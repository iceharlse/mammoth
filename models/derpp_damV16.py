import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models import register_model

# ============================================================================
# 1. Controller (Standard V12/V14 Logic)
# ============================================================================
class AlphaBetaControllerV16(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        alpha_min: float = 0.60, 
        alpha_max: float = 0.85,
        num_layers_out: int = 25,
    ):
        super().__init__()
        self.ctx_dim = 8
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.mlp_ctx = nn.Sequential(
            nn.Linear(self.ctx_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.project_feature = nn.Linear(feature_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head_p_stable = nn.Linear(d_model, 1)

        self.head_beta = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_layers_out)
        )
        nn.init.constant_(self.head_beta[2].bias, -1.0) 

    def forward(self, ctx, mu_old, mu_new):
        ctx_emb = self.mlp_ctx(ctx)
        old_emb = self.project_feature(mu_old)
        new_emb = self.project_feature(mu_new)

        tokens = torch.stack([ctx_emb, old_emb, new_emb], dim=1)
        out = self.transformer(tokens)
        h_ctx = out[:, 0, :]

        p_stable = torch.sigmoid(self.head_p_stable(h_ctx)).squeeze()
        w_old = p_stable * self.alpha_max + (1.0 - p_stable) * self.alpha_min
        w_new = 1.0 - w_old

        beta_logits = self.head_beta(h_ctx).squeeze()
        beta_profile = torch.sigmoid(beta_logits)

        return w_old, w_new, p_stable, beta_profile


# ============================================================================
# 2. DER++ + DAM V16 (Orthogonal Integration)
#    - MSE Loss (DER++): Passed through directly (Static Protection)
#    - CE Losses (New + Replay): Managed by DAM (Dynamic Modulation)
# ============================================================================
@register_model("derpp_dam_v16")
class DERpp_DAM_V16(ContinualModel):
    NAME = "derpp_dam_v16"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        add_rehearsal_args(parser)
        
        # DER++ Args
        parser.add_argument('--der_alpha', type=float, default=0.2, help='DER++ MSE Weight')
        parser.add_argument('--der_beta', type=float, default=0.5, help='DER++ CE Weight (Not used directly, managed by DAM)')

        # DAM Args
        parser.add_argument("--dam_d_model", type=int, default=64)
        parser.add_argument("--dam_nhead", type=int, default=4)
        parser.add_argument("--dam_layers", type=int, default=2)
        
        parser.add_argument("--meta_lr", type=float, default=5e-4)
        parser.add_argument("--meta_interval", type=int, default=20)
        parser.add_argument("--w_reg_strength", type=float, default=0.01)
        parser.add_argument("--beta_meta_coef", type=float, default=1.0)
        
        parser.add_argument("--alpha_min", type=float, default=0.60)
        parser.add_argument("--alpha_max", type=float, default=0.85)
        parser.add_argument("--ctx_ema_beta", type=float, default=0.9)
        
        parser.add_argument("--mgda_blend", type=float, default=0.5)
        
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        if hasattr(self.net, "num_features"):
            self.feature_dim = self.net.num_features
        else:
            self.feature_dim = 512

        self.layer_names = []
        for name, param in self.net.named_parameters():
            if param.requires_grad and ("weight" in name) and ("bn" not in name):
                layer_id = ".".join(name.split(".")[:-1])
                if layer_id not in self.layer_names:
                    self.layer_names.append(layer_id)
        
        self.num_layers_out = max(len(self.layer_names), 1)

        self.controller = AlphaBetaControllerV16(
            feature_dim=self.feature_dim,
            d_model=self.args.dam_d_model,
            num_layers_out=self.num_layers_out,
            alpha_min=getattr(self.args, "alpha_min", 0.60),
            alpha_max=getattr(self.args, "alpha_max", 0.85),
        ).to(self.device)

        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)
        self.opt_cont = torch.optim.Adam(self.controller.parameters(), lr=self.args.meta_lr, weight_decay=1e-5)
        
        self.global_step = 0
        self.current_task_id = 0
        self.ctx_ema = None

    def _get_features(self, x):
        return self.net(x, returnt="features")

    def _get_layer_groups(self):
        groups = {name: [] for name in self.layer_names}
        for name, param in self.net.named_parameters():
            if not param.requires_grad: continue
            for key in self.layer_names:
                if name.startswith(key):
                    groups[key].append(param)
                    break
        return groups

    def _build_ctx(self, loss_new, loss_old, feats_new, feats_old):
        # V12 Log-Loss Logic
        l_n_val, l_o_val = loss_new.item(), loss_old.item()
        log_l_n = math.log1p(l_n_val) 
        log_l_o = math.log1p(l_o_val)
        denom = log_l_n + log_l_o + 1e-8
        
        ctx = torch.tensor([
            log_l_o / denom, log_l_n / denom, log_l_n - log_l_o, log_l_n + log_l_o,
            self.current_task_id / 5.0, 0, 0, 0 
        ]).to(self.device).float().unsqueeze(0)
        
        mu_n = feats_new.mean(dim=0, keepdim=True).detach()
        mu_o = feats_old.mean(dim=0, keepdim=True).detach()
        return ctx, mu_n, mu_o

    # MGDA Solver
    def _solve_mgda(self, grads_new, grads_old):
        gn_list = [g.view(-1) for g in grads_new if g is not None]
        go_list = [g.view(-1) for g in grads_old if g is not None]
        if not gn_list or not go_list: return 0.5
        gn = torch.cat(gn_list)
        go = torch.cat(go_list)
        dot_oo = torch.dot(go, go)
        dot_nn = torch.dot(gn, gn)
        dot_on = torch.dot(gn, go)
        den = dot_oo + dot_nn - 2*dot_on
        if den < 1e-8: return 0.5
        alpha = (dot_nn - dot_on) / den
        return torch.clamp(alpha, 0.0, 1.0)

    # ----------------------------------------------------------------------
    # Observation Loop (Orthogonal Integration)
    # ----------------------------------------------------------------------
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.global_step += 1
        
        self.opt.zero_grad()
        self.net.train()
        self.controller.eval()

        # 1. New Task CE
        feats_new = self._get_features(inputs)
        out_new = self.net.classifier(feats_new)
        loss_new_ce = self.loss(out_new, labels)
        
        if self.buffer.is_empty():
            loss_new_ce.backward()
            self.opt.step()
            self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=out_new.data)
            return loss_new_ce.item()

        # 2. Replay Data (for both CE and MSE)
        # Note: No return_logits arg needed, assume buffer returns 3 items
        buf_ret = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
        if len(buf_ret) == 2: buf_inputs, buf_labels = buf_ret; buf_logits = None
        else: buf_inputs, buf_labels, buf_logits = buf_ret

        feats_old = self._get_features(buf_inputs)
        out_old = self.net.classifier(feats_old)
        
        # 3. Old Task CE (Managed by DAM)
        loss_old_ce = self.loss(out_old, buf_labels)

        # 4. Old Task MSE (Direct Add, Not managed by DAM)
        # We compute gradients for MSE separately
        loss_mse = torch.tensor(0.0).to(self.device)
        if buf_logits is not None:
            loss_mse = self.args.der_alpha * F.mse_loss(out_old, buf_logits)

        # -----------------------------------------------------------
        # PART A: DAM Management (New CE vs Old CE)
        # -----------------------------------------------------------
        with torch.no_grad():
            ctx, mu_n, mu_o = self._build_ctx(loss_new_ce, loss_old_ce, feats_new, feats_old)
            if self.ctx_ema is None: self.ctx_ema = ctx
            else: self.ctx_ema = self.args.ctx_ema_beta * self.ctx_ema + (1 - self.args.ctx_ema_beta) * ctx
            w_old_meta, w_new_meta, p_stable, beta_profile = self.controller(self.ctx_ema, mu_o, mu_n)

        # Gradients for CE parts
        params = [p for p in self.net.parameters() if p.requires_grad]
        grads_new_ce = torch.autograd.grad(loss_new_ce, params, retain_graph=True, allow_unused=True)
        grads_old_ce = torch.autograd.grad(loss_old_ce, params, retain_graph=True, allow_unused=True) # Retain for MSE next

        # Calculate MGDA for CE parts
        alpha_mgda = self._solve_mgda(grads_new_ce, grads_old_ce)
        eta = self.args.mgda_blend
        w_old_final = (1 - eta) * w_old_meta + eta * alpha_mgda
        w_new_final = 1.0 - w_old_final

        p_to_gn = {p: g for p, g in zip(params, grads_new_ce)}
        p_to_go = {p: g for p, g in zip(params, grads_old_ce)}

        # -----------------------------------------------------------
        # PART B: Apply DAM to CE + Add MSE
        # -----------------------------------------------------------
        
        # Calculate MSE Gradients independently
        grads_mse = torch.autograd.grad(loss_mse, params, retain_graph=False, allow_unused=True)
        p_to_gmse = {p: g for p, g in zip(params, grads_mse)}

        layer_groups = self._get_layer_groups()
        for p in params: p.grad = None
        
        for i, (layer_name, layer_params) in enumerate(layer_groups.items()):
            if not layer_params: continue
            
            beta_l = beta_profile[i].item() if i < len(beta_profile) else beta_profile[-1].item()
            
            # Gradient Matching (CE vs CE only)
            gn_norm, go_norm = 0.0, 0.0
            for p in layer_params:
                gn = p_to_gn.get(p)
                go = p_to_go.get(p)
                if gn is not None: gn_norm += gn.norm().item()**2
                if go is not None: go_norm += go.norm().item()**2
            gn_norm = math.sqrt(gn_norm)
            go_norm = math.sqrt(go_norm)
            
            if gn_norm > 1e-8 and go_norm > 1e-8:
                scaler = go_norm / gn_norm
                scaler = min(scaler, 1.0)
            else:
                scaler = 1.0

            for p in layer_params:
                # 1. DAM Managed: New CE vs Old CE
                gn = p_to_gn.get(p)
                go = p_to_go.get(p)
                if gn is None: gn = torch.zeros_like(p)
                if go is None: go = torch.zeros_like(p)

                grad_dam = w_old_final * go + (w_new_final * beta_l) * (gn * scaler)
                
                # 2. Static Managed: MSE (Add directly)
                gmse = p_to_gmse.get(p)
                if gmse is None: gmse = torch.zeros_like(p)
                
                # Final Gradient = DAM(CE) + MSE
                p.grad = grad_dam + gmse

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        
        self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=out_new.data)

        if self.global_step % self.args.meta_interval == 0:
            self._meta_update(inputs, labels, buf_inputs, buf_labels) # Meta only looks at CE

        # Return total loss for logging
        return loss_new_ce.item() + loss_old_ce.item() + loss_mse.item()

    def _meta_update(self, val_inputs, val_labels, buf_inputs, buf_labels):
        # Meta Update only trains on the CE trade-off logic
        # We assume MSE is a static constraint we don't meta-learn against
        self.net.eval()
        self.controller.train()
        self.opt_cont.zero_grad()

        feats_new = self._get_features(val_inputs)
        out_new = self.net.classifier(feats_new)
        loss_new = self.loss(out_new, val_labels)
        
        feats_old = self._get_features(buf_inputs)
        out_old = self.net.classifier(feats_old)
        loss_old = self.loss(out_old, buf_labels)

        ctx, mu_n, mu_o = self._build_ctx(loss_new, loss_old, feats_new, feats_old)
        w_old, w_new, p_stable, beta_profile = self.controller(ctx, mu_o, mu_n)

        with torch.no_grad():
            l_o, l_n = loss_old.item(), loss_new.item()
            target_w_old = l_o / (l_o + l_n + 1e-8)
            target_w_old = max(0.2, min(0.8, target_w_old))
        loss_alpha = F.mse_loss(w_old, torch.tensor(target_w_old).to(self.device))

        # Beta Ratio Logic (CE vs CE)
        params = [p for p in self.net.parameters() if p.requires_grad]
        g_meta_new = torch.autograd.grad(loss_new, params, retain_graph=True, allow_unused=True)
        g_meta_old = torch.autograd.grad(loss_old, params, retain_graph=False, allow_unused=True)
        p_to_idx = {id(p): idx for idx, p in enumerate(params)}
        layer_groups = self._get_layer_groups()
        
        beta_targets = []
        for layer_name in self.layer_names:
            p_list = layer_groups[layer_name]
            if not p_list: 
                beta_targets.append(0.5)
                continue
            gn_sq_sum, go_sq_sum = 0.0, 0.0
            for p in p_list:
                idx = p_to_idx.get(id(p))
                if idx is None: continue
                gn, go = g_meta_new[idx], g_meta_old[idx]
                if gn is not None: gn_sq_sum += gn.norm().item()**2
                if go is not None: go_sq_sum += go.norm().item()**2
            denom = gn_sq_sum + go_sq_sum + 1e-8
            beta_star = gn_sq_sum / denom
            beta_targets.append(beta_star)
            
        if beta_targets:
            L = min(len(beta_targets), beta_profile.shape[0])
            loss_beta = F.mse_loss(beta_profile[:L], torch.tensor(beta_targets[:L]).to(self.device))
        else:
            loss_beta = torch.tensor(0.0).to(self.device)

        loss_reg = self.args.w_reg_strength * (p_stable - 0.5)**2
        total_meta_loss = loss_alpha + loss_reg + self.args.beta_meta_coef * loss_beta
        total_meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 1.0)
        self.opt_cont.step()