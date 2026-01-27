# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini,
# Angelo Porrello, Simone Calderara.
# All rights reserved.

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.batch_norm import bn_track_stats
from utils.buffer import Buffer
from utils import binary_to_boolean_type, none_or_float
from models import register_model

# STAR Imports
from models.star_utils.star_perturber import Perturber, add_perturb_args


def dsimplex(num_classes=10):
    """Simplex coordinates used by RPC head."""
    def simplex_coordinates2(m):
        x = np.zeros([m, m + 1])
        for j in range(0, m):
            x[j, j] = 1.0

        a = (1.0 - np.sqrt(float(1 + m))) / float(m)
        for i in range(0, m):
            x[i, m] = a

        c = np.zeros(m)
        for i in range(0, m):
            s = 0.0
            for j in range(0, m + 1):
                s = s + x[i, j]
            c[i] = s / float(m + 1)

        for j in range(0, m + 1):
            for i in range(0, m):
                x[i, j] = x[i, j] - c[i]

        s = 0.0
        for i in range(0, m):
            s = s + x[i, 0] ** 2
        s = np.sqrt(s)

        for j in range(0, m + 1):
            for i in range(0, m):
                x[i, j] = x[i, j] / s

        return x

    feat_dim = num_classes - 1
    ds = simplex_coordinates2(feat_dim)
    return ds


class AlphaBetaController(nn.Module):
    def __init__(self, ctx_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(ctx)


@register_model("xder_rpc_star_dam")
class XDerRPCStarDam(ContinualModel):
    NAME = 'xder_rpc_star_dam'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        add_rehearsal_args(parser)
        add_perturb_args(parser)

        # XDER Base Args
        parser.add_argument('--alpha', type=float, required=True, help='Distillation (MSE) weight.')
        parser.add_argument('--beta', type=float, required=True, help='Replay (CE) weight.')
        parser.add_argument('--gamma', type=float, default=0.85)
        parser.add_argument('--constr_eta', type=float, default=0.1)
        parser.add_argument('--constr_margin', type=float, default=0.3)
        parser.add_argument('--clip_grad', type=none_or_float, default=None, metavar='NORM')
        parser.add_argument('--align_bn', type=binary_to_boolean_type, default=0, help='Use BatchNorm alignment')
        parser.add_argument('--n_rpc_heads', type=int, help='N Heads for RPC')

        # DAM Args
        parser.add_argument('--dam_alpha_gain', type=float, default=0.0)
        parser.add_argument('--dam_beta_gain', type=float, default=0.0)
        parser.add_argument('--dam_hidden', type=int, default=32)
        parser.add_argument('--dam_reg', type=float, default=1e-3)
        
        # DAM: Beta Health
        parser.add_argument('--dam_beta_health', type=int, default=0)
        parser.add_argument('--dam_beta_eta', type=float, default=0.01)
        parser.add_argument('--dam_beta_c', type=float, default=0.1)
        parser.add_argument('--dam_beta_rho', type=float, default=0.2)
        
        # DAM: Cone
        parser.add_argument('--dam_cone_cos', type=float, default=0.0)
        parser.add_argument('--dam_cone_norm_eps', type=float, default=0.3)
        
        parser.add_argument('--num_tasks', type=int, default=10)

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size)
        
        self.register_buffer('update_counter', torch.zeros(self.args.buffer_size))

        n_rpc_heads = self.args.n_rpc_heads if self.args.n_rpc_heads is not None else self.num_classes
        self.register_buffer('rpc_head', torch.from_numpy(dsimplex(n_rpc_heads)).float())

        if not hasattr(self.args, 'start_from'):
            self.args.start_from = 0

        self.pert = Perturber(self)

        self.ctx_dim = 8
        self.controller = AlphaBetaController(self.ctx_dim, self.args.dam_hidden).to(self.device)
        self.opt_ctrl = torch.optim.Adam(self.controller.parameters(), lr=self.args.lr, weight_decay=1e-5)
        self.grad_ema = {}

    def forward(self, x):
        x = self.net(x)[:, :-1]
        if x.dtype != self.rpc_head.dtype:
            self.rpc_head = self.rpc_head.type(x.dtype)
        if self.rpc_head.device != x.device:
            self.rpc_head = self.rpc_head.to(x.device)
            
        x = x @ self.rpc_head
        return x

    def end_task(self, dataset):
        was_training = self.training
        self.train()

        if self.args.start_from is None or self.current_task >= self.args.start_from:
            # Reduce Memory Buffer
            if self.current_task > 0:
                examples_per_class = self.args.buffer_size // self.n_seen_classes
                buf_x, buf_lab, buf_log, buf_tl = self.buffer.get_all_data(device=self.device)
                self.buffer.empty()
                for tl in buf_lab.unique():
                    idx = tl == buf_lab
                    ex, lab, log, tasklab = buf_x[idx], buf_lab[idx], buf_log[idx], buf_tl[idx]
                    first = min(ex.shape[0], examples_per_class)
                    self.buffer.add_data(
                        examples=ex[:first],
                        labels=lab[:first],
                        logits=log[:first],
                        task_labels=tasklab[:first]
                    )

            # Add new task data
            examples_last_task = self.buffer.buffer_size - self.buffer.num_seen_examples
            examples_per_class = examples_last_task // self.n_classes_current_task
            ce = torch.tensor([examples_per_class] * self.n_classes_current_task).int()
            ce[torch.randperm(self.n_classes_current_task)[:examples_last_task - (examples_per_class * self.n_classes_current_task)]] += 1

            with torch.no_grad():
                with bn_track_stats(self, False):
                    if self.args.start_from is None or self.args.start_from <= self.current_task:
                        for data in dataset.train_loader:
                            inputs, labels, not_aug_inputs = data[0], data[1], data[2]
                            inputs = inputs.to(self.device)
                            not_aug_inputs = not_aug_inputs.to(self.device)
                            outputs = self(inputs)
                            if all(ce == 0):
                                break

                            if self.current_task > 0:
                                outputs = self.update_logits(outputs, outputs, labels, 0, self.current_task)

                            flags = torch.zeros(len(inputs)).bool()
                            for j in range(len(flags)):
                                if ce[labels[j] % self.n_classes_current_task] > 0:
                                    flags[j] = True
                                    ce[labels[j] % self.n_classes_current_task] -= 1

                            self.buffer.add_data(examples=not_aug_inputs[flags],
                                                 labels=labels[flags],
                                                 logits=outputs.data[flags],
                                                 task_labels=(torch.ones(len(not_aug_inputs)) * self.current_task)[flags])

                    # Update future past logits
                    buf_idx, buf_inputs, buf_labels, buf_logits, _ = self.buffer.get_data(
                        self.buffer.buffer_size, transform=self.transform, return_index=True, device=self.device)

                    buf_outputs = []
                    while len(buf_inputs):
                        buf_outputs.append(self(buf_inputs[:self.args.batch_size]))
                        buf_inputs = buf_inputs[self.args.batch_size:]
                    buf_outputs = torch.cat(buf_outputs)

                    chosen = ((buf_labels // self.n_classes_current_task) < self.current_task).to(self.buffer.device)

                    if chosen.any():
                        to_transplant = self.update_logits(buf_logits[chosen], buf_outputs[chosen], buf_labels[chosen],
                                                          self.current_task, self.n_tasks - self.current_task)
                        
                        buf_device = self.buffer.logits.device
                        idx_to_update = buf_idx[chosen].to(buf_device)
                        self.buffer.logits[idx_to_update, :] = to_transplant.to(buf_device)
                        self.buffer.task_labels[idx_to_update] = self.current_task

        self.update_counter.zero_()
        self.train(was_training)

    def update_logits(self, old, new, gt, task_start, n_tasks=1):
        offset_1, _ = self.dataset.get_offsets(task_start)
        offset_2, _ = self.dataset.get_offsets(task_start + n_tasks)
        transplant = new[:, offset_1:offset_2]
        gt_values = old[torch.arange(len(gt)), gt]
        max_values = transplant.max(1).values
        coeff = self.args.gamma * gt_values / max_values
        coeff = coeff.unsqueeze(1).repeat(1, offset_2 - offset_1)
        mask = (max_values > gt_values).unsqueeze(1).repeat(1, offset_2 - offset_1)
        transplant[mask] *= coeff[mask]
        old[:, offset_1:offset_2] = transplant
        return old

    def _select_head_params(self):
        for attr in ["classifier", "fc", "head", "last"]:
            if hasattr(self.net, attr):
                mod = getattr(self.net, attr)
                if hasattr(mod, "parameters"):
                    ps = [p for p in mod.parameters() if p.requires_grad]
                    if ps: return ps
        linear_params = []
        for n, m in self.net.named_modules():
            if isinstance(m, nn.Linear):
                linear_params = [p for p in m.parameters() if p.requires_grad]
        if linear_params: return linear_params
        return []

    def _compute_head_grad_stats(self, loss_new: torch.Tensor, loss_old_ce: torch.Tensor):
        head_params = self._select_head_params()
        if not head_params:
            return 0.0, 0.0, 0.0

        g_old = torch.autograd.grad(loss_old_ce, head_params, retain_graph=True, allow_unused=True)
        g_new = torch.autograd.grad(loss_new, head_params, retain_graph=True, allow_unused=True)

        g_old_flat = torch.cat([g.reshape(-1) for g in g_old if g is not None], dim=0) if any(g is not None for g in g_old) else None
        g_new_flat = torch.cat([g.reshape(-1) for g in g_new if g is not None], dim=0) if any(g is not None for g in g_new) else None
        
        if g_old_flat is None or g_new_flat is None or g_old_flat.numel() == 0 or g_new_flat.numel() == 0:
            return 0.0, 0.0, 0.0

        eps = 1e-8
        norm_old = g_old_flat.norm() + eps
        norm_new = g_new_flat.norm() + eps
        cos_theta = (g_old_flat @ g_new_flat) / (norm_old * norm_new)
        cos_theta = cos_theta.clamp(-1.0, 1.0)
        log_old = torch.log(norm_old) / 5.0
        log_new = torch.log(norm_new) / 5.0
        return float(cos_theta.detach().item()), float(log_old.detach().item()), float(log_new.detach().item())

    def _build_context(self, loss_new, loss_old_ce, loss_old_mse, cos_theta, log_g_old, log_g_new):
        alpha0 = float(self.args.alpha)
        beta0 = float(self.args.beta)
        l_new = float(loss_new.detach().item())
        l_old = float((beta0 * loss_old_ce + alpha0 * loss_old_mse).detach().item())

        log_l_n = math.log1p(max(l_new, 0.0))
        log_l_o = math.log1p(max(l_old, 0.0))
        denom = log_l_n + log_l_o + 1e-8

        num_tasks = float(max(getattr(self.args, "num_tasks", 10), 1))
        cur_task = getattr(self, "current_task", 0)
        try:
            task_prog = float(cur_task) / max(num_tasks - 1.0, 1.0)
        except Exception:
            task_prog = 0.0

        ctx = torch.tensor(
            [log_l_o / denom, log_l_n / denom, log_l_n - log_l_o, log_l_n + log_l_o,
             task_prog, float(cos_theta), float(log_g_old), float(log_g_new)],
            device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        return ctx

    def _beta_grad_health_update(self, c_eff, rho_eff):
        if getattr(self.args, "dam_beta_health", 0) <= 0: return
        eta = float(getattr(self.args, "dam_beta_eta", 0.01))
        c, rho = float(c_eff), float(max(rho_eff, 0.0))
        eps = 1e-8

        for name, p in self.net.named_parameters():
            if p.grad is None: continue
            g = p.grad.data.norm(2)
            if not torch.isfinite(g): continue
            
            if name not in self.grad_ema:
                self.grad_ema[name] = g.detach()
                continue

            ema = self.grad_ema[name]
            new_ema = (1.0 - eta) * ema + eta * g
            self.grad_ema[name] = new_ema.detach()

            r = g / (new_ema + eps)
            scale = torch.clamp(1.0 + c * (1.0 - r), min=1.0 - rho, max=1.0 + rho)
            p.grad.data.mul_(scale)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.net.train()
        self.controller.train()
        self.opt.zero_grad()
        self.opt_ctrl.zero_grad()
        
        net_params = [p for p in self.net.parameters() if p.requires_grad]
        ctrl_params = [p for p in self.controller.parameters() if p.requires_grad]

        # =========================================================
        # 1. STAR Step
        # =========================================================
        g_star_list = None
        if not self.buffer.is_empty():
            buf_ret = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_inputs_star, buf_labels_star = buf_ret[0], buf_ret[1]
            
            self.pert(buf_inputs_star, buf_labels_star)

            g_star_list = []
            for p in net_params:
                if p.grad is None:
                    g_star_list.append(torch.zeros_like(p))
                else:
                    g_star_list.append(p.grad.detach().clone())
            self.opt.zero_grad()

        # =========================================================
        # 2. XDER-RPC Losses
        # =========================================================
        outputs = self(inputs)
        loss_stream = self.loss(outputs[:, self.n_past_classes:self.n_seen_classes],
                                labels - self.n_past_classes)

        loss_constr_futu = torch.tensor(0., device=self.device, dtype=loss_stream.dtype)
        if self.current_task < self.n_tasks - 1:
            bad_head = outputs[:, self.n_seen_classes:]
            good_head = outputs[:, self.n_past_classes:self.n_seen_classes]
            loss_constr = bad_head.max(1)[0] + self.args.constr_margin - good_head.max(1)[0]
            mask = loss_constr > 0
            if mask.any():
                loss_constr_futu = self.args.constr_eta * loss_constr[mask].mean()
        
        loss_new = loss_stream + loss_constr_futu

        if self.buffer.is_empty():
            loss_new.backward()
            if g_star_list is not None: 
                for i, p in enumerate(net_params):
                     if p.grad is not None: p.grad.add_(g_star_list[i])

            if self.args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip_grad)
            self.opt.step()
            # [FIX] Do NOT add data to buffer here for XDER
            return loss_new.item()

        # Buffer Losses
        loss_old_mse = torch.tensor(0., device=self.device, dtype=loss_stream.dtype)
        loss_old_ce = torch.tensor(0., device=self.device, dtype=loss_stream.dtype)

        # Distillation (MSE)
        buf_idx1, buf_inputs1, buf_labels1, buf_logits1, _ = self.buffer.get_data(
            self.args.minibatch_size, transform=self.transform, return_index=True, device=self.device)
        with bn_track_stats(self, False):
            buf_outputs1 = self(buf_inputs1)
        buf_logits1 = buf_logits1.type(buf_outputs1.dtype)
        loss_old_mse = F.mse_loss(buf_outputs1, buf_logits1)

        # Replay (CE)
        buf_idx2, buf_inputs2, buf_labels2, buf_logits2, _ = self.buffer.get_data(
            self.args.minibatch_size, transform=self.transform, return_index=True, device=self.device)
        with bn_track_stats(self, False):
            buf_outputs2 = self(buf_inputs2).float()
        loss_old_ce = self.loss(buf_outputs2[:, :self.n_past_classes], buf_labels2)

        # Buffer update logic (Logits update prep)
        buf_idx = torch.cat([buf_idx1, buf_idx2])
        buf_labels = torch.cat([buf_labels1, buf_labels2])
        buf_outputs = torch.cat([buf_outputs1, buf_outputs2])
        buf_logits = torch.cat([buf_logits1, buf_logits2])
        
        with torch.no_grad():
            chosen = ((buf_labels // self.n_classes_current_task) < self.current_task).to(self.device)
            
            if self.update_counter.device != self.device:
                self.update_counter = self.update_counter.to(self.device)
            
            self.update_counter[buf_idx[chosen]] += 1
            c = chosen.clone()
            
            chosen[c] = torch.rand_like(chosen[c].float()) * self.update_counter[buf_idx[c]] < 1
            
            if chosen.any():
                to_transplant = self.update_logits(
                    buf_logits[chosen], buf_outputs[chosen], buf_labels[chosen],
                    self.current_task, self.n_tasks - self.current_task
                ).to(self.device)
                
                buf_device = self.buffer.logits.device
                idx_to_update = buf_idx[chosen].to(buf_device)
                transplant_data = to_transplant.to(buf_device)
                
                self.buffer.logits[idx_to_update, :] = transplant_data
                self.buffer.task_labels[idx_to_update] = self.current_task

        # =========================================================
        # 3. DAM
        # =========================================================
        cos_t, log_go, log_gn = self._compute_head_grad_stats(loss_new, loss_old_ce)
        with torch.no_grad():
            ctx = self._build_context(loss_new, loss_old_ce, loss_old_mse, cos_t, log_go, log_gn)
        
        delta = self.controller(ctx).squeeze(0)
        delta_alpha, delta_beta = delta[0], delta[1]
        
        alpha0, beta0 = float(self.args.alpha), float(self.args.beta)
        gain_a, gain_b = float(self.args.dam_alpha_gain), float(self.args.dam_beta_gain)
        
        alpha_eff = alpha0 * (1.0 + gain_a * delta_alpha)
        beta_eff = beta0 * (1.0 + gain_b * delta_beta)
        
        alpha_eff = torch.clamp(alpha_eff, min=alpha0*(1-gain_a), max=alpha0*(1+gain_a))
        beta_eff = torch.clamp(beta_eff, min=beta0*(1-gain_b), max=beta0*(1+gain_b))
        
        beta_factor = torch.clamp(1.0 + gain_b * delta_beta, min=1-gain_b, max=1+gain_b)

        loss_anchor = loss_new + alpha0 * loss_old_mse + beta0 * loss_old_ce
        loss_total = loss_new + alpha_eff * loss_old_mse + beta_eff * loss_old_ce
        
        if self.args.dam_reg > 0.0:
            loss_total += self.args.dam_reg * (delta_alpha**2 + delta_beta**2)

        g_anchor = torch.autograd.grad(loss_anchor, net_params, retain_graph=True, allow_unused=True)
        g_total = torch.autograd.grad(loss_total, net_params, retain_graph=True, allow_unused=True)
        
        flat_anchor = torch.cat([g.view(-1) if g is not None else torch.zeros_like(p).view(-1) for g, p in zip(g_anchor, net_params)])
        flat_total = torch.cat([g.view(-1) if g is not None else torch.zeros_like(p).view(-1) for g, p in zip(g_total, net_params)])

        # Cone Projection
        eps = 1e-8
        norm_a = flat_anchor.norm() + eps
        norm_t = flat_total.norm() + eps
        
        if norm_a < 1e-12:
            flat_safe = flat_total
        else:
            cos_theta = torch.dot(flat_total, flat_anchor) / (norm_t * norm_a)
            cos_min = float(self.args.dam_cone_cos)
            
            if cos_theta.item() < cos_min:
                proj_coeff = torch.dot(flat_total, flat_anchor) / (norm_a * norm_a)
                flat_safe = proj_coeff * flat_anchor
            else:
                flat_safe = flat_total
            
            max_norm = (1.0 + float(self.args.dam_cone_norm_eps)) * norm_a
            if flat_safe.norm() > max_norm:
                flat_safe = flat_safe * (max_norm / (flat_safe.norm() + eps))

        offset = 0
        for i, p in enumerate(net_params):
            numel = p.numel()
            g_safe_chunk = flat_safe[offset : offset+numel].view_as(p)
            
            if g_star_list is not None:
                g_safe_chunk = g_safe_chunk + g_star_list[i]
            
            p.grad = g_safe_chunk.detach().clone()
            offset += numel

        c_eff = float(self.args.dam_beta_c) * float(beta_factor.detach())
        rho_eff = float(self.args.dam_beta_rho) * float(beta_factor.detach())
        self._beta_grad_health_update(c_eff, rho_eff)
        
        if ctrl_params:
            g_ctrl = torch.autograd.grad(loss_total, ctrl_params, retain_graph=False, allow_unused=True)
            for p, g in zip(ctrl_params, g_ctrl):
                if g is not None: p.grad = g.detach().clone()

        if self.args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip_grad)
        
        self.opt.step()
        self.opt_ctrl.step()

        # [FIX] Do NOT add data to buffer here. XDER fills buffer in end_task.
        # self.buffer.add_data(examples=not_aug_inputs, labels=labels)
        
        return loss_total.item()