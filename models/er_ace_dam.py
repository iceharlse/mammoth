import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils import binary_to_boolean_type
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models import register_model


class CustomLinear(torch.nn.Module):
    """
    Same custom classifier as the original ER-ACE implementation:
    normalized features, normalized weights, cosine scores scaled by a factor.
    """
    def __init__(self, indim, outdim, weight=None):
        super(CustomLinear, self).__init__()
        self.L = torch.nn.Linear(indim, outdim, bias=False)
        if weight is not None:
            self.L.weight.data = weight.clone()

        self.scale_factor = 10

    def forward(self, x: torch.Tensor):
        # normalize feature
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)

        # normalize weights
        w = self.L.weight
        w_norm = torch.norm(w, p=2, dim=1).unsqueeze(1).expand_as(w)
        cos_dist = torch.mm(
            x_normalized,
            w.div(w_norm + 1e-5).transpose(0, 1)
        )
        scores = self.scale_factor * cos_dist
        return scores


class AlphaBetaController(nn.Module):
    """
    Small MLP controller.

    Input  ctx = [log L_new, log L_buf_ce, log L_buf_mse, task_progress]
    Output [Δα_raw, Δβ_raw] ∈ [-1, 1]^2
    """
    def __init__(self, ctx_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),  # outputs in [-1, 1]
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(ctx)


@register_model("er_ace_dam")
class ErACEDamAlphaBetaConstrainedV3(ContinualModel):
    """
    ER-ACE + α/β-controller + gradient cone + β-grad-health,
    with α controlling an explicit MSE distillation term on buffer examples.

    Design for ER-ACE:
      - anchor = original ER-ACE loss: L_anchor = L_new + beta * L_buf_ce
      - total  = L_total = L_new + beta_eff * L_buf_ce + lambda_mse * L_buf_mse + reg

      - alpha-branch:
            lambda_mse ∈ [0, alpha], learned by controller
      - beta-branch:
            beta_eff = beta * (1 + dam_beta_gain * Δβ_raw), Δβ_raw ∈ [-1,1]
            then clamped to [beta*(1-dam_beta_cap), beta*(1+dam_beta_cap)]
      - cone:
            g_total is projected into a cone around g_anchor:
              cos(g_total, g_anchor) ≥ dam_cone_cos
              ||g_total|| ≤ (1 + dam_cone_norm_eps) ||g_anchor||

      - β-grad-health:
            small per-parameter rescaling based on grad_norm / EMA,
            with strength ∝ |Δβ_raw| * task_progress.

    When alpha=0, dam_alpha_gain=0, dam_beta_gain=0, dam_beta_health=0,
    dam_cone_cos<=-1, dam_cone_norm_eps>=999, dam_reg=0, this recovers ER-ACE.
    """

    NAME = "er_ace_dam"
    COMPATIBILITY = ["class-il", "task-il"]

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        add_rehearsal_args(parser)

        parser.add_argument(
            "--task_free",
            type=binary_to_boolean_type,
            default=False,
            help="Enable task-free training (replay starts from second task)?",
        )
        parser.add_argument(
            "--use_custom_classifier",
            type=binary_to_boolean_type,
            default=True,
            help="Use the custom classifier used in the original ER-ACE work.",
        )

        parser.add_argument(
            "--num_tasks",
            type=int,
            default=10,
            help="Total number of tasks in the scenario (for task-progress scheduling).",
        )

        # alpha: max weight for MSE distillation on buffer
        parser.add_argument(
            "--alpha",
            type=float,
            default=0.0,
            help=(
                "Max weight for MSE distillation on buffer logits.\n"
                "Effective lambda_mse is in [0, alpha], learned by controller.\n"
                "Set 0 to disable the MSE branch."
            ),
        )

        # beta: base replay weight (ER-ACE uses 1.0)
        parser.add_argument(
            "--beta",
            type=float,
            default=1.0,
            help="Base weight for buffer CE replay (ER-ACE uses 1.0).",
        )

        parser.add_argument(
            "--dam_alpha_gain",
            type=float,
            default=1.0,
            help="Gain for alpha-controller. Only used if alpha>0.",
        )
        parser.add_argument(
            "--dam_beta_gain",
            type=float,
            default=0.0,
            help=(
                "Gain for beta-controller. Effective beta:\n"
                "  beta_eff = beta * (1 + dam_beta_gain * Δβ_raw), Δβ_raw ∈ [-1,1]\n"
                "then clamped to [beta*(1-dam_beta_cap), beta*(1+dam_beta_cap)]."
            ),
        )
        parser.add_argument(
            "--dam_beta_cap",
            type=float,
            default=0.2,
            help="Max relative change for beta_eff around beta (e.g., 0.2 → ±20%).",
        )
        parser.add_argument(
            "--dam_hidden",
            type=int,
            default=32,
            help="Hidden dim of alpha/beta controller MLP.",
        )
        parser.add_argument(
            "--dam_reg",
            type=float,
            default=0.0,
            help="L2 regularization on controller outputs (Δα_raw, Δβ_raw).",
        )

        # β-grad-health
        parser.add_argument(
            "--dam_beta_health",
            type=int,
            default=0,
            help="If >0, enable beta grad-health doctor.",
        )
        parser.add_argument(
            "--dam_beta_eta",
            type=float,
            default=0.01,
            help="EMA momentum for grad-health.",
        )
        parser.add_argument(
            "--dam_beta_c",
            type=float,
            default=0.05,
            help="Base strength for grad-health.",
        )
        parser.add_argument(
            "--dam_beta_rho",
            type=float,
            default=0.1,
            help="Base max multiplicative change per step for grad-health.",
        )

        # gradient cone
        parser.add_argument(
            "--dam_cone_cos",
            type=float,
            default=0.0,
            help="Minimum cosine between DAM grad and ER-ACE anchor; "
                 "0≈90°, >0 stronger alignment. Use -1 to disable.",
        )
        parser.add_argument(
            "--dam_cone_norm_eps",
            type=float,
            default=0.2,
            help="Norm constraint: ||g_safe|| ≤ (1+eps)||g_anchor||. "
                 "Use a very large value (e.g. 999) to effectively disable.",
        )

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if args.use_custom_classifier:
            assert hasattr(backbone, "classifier"), "Backbone must have a classifier layer."
            backbone.classifier = CustomLinear(
                backbone.classifier.in_features,
                backbone.classifier.out_features,
            )
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size)
        self.seen_so_far = torch.tensor([], device=self.device).long()

        # controller
        self.ctx_dim = 4  # [log L_new, log L_buf_ce, log L_buf_mse, task_progress]
        self.controller = AlphaBetaController(
            ctx_dim=self.ctx_dim,
            hidden_dim=self.args.dam_hidden,
        ).to(self.device)

        self.opt_ctrl = torch.optim.Adam(
            (p for p in self.controller.parameters() if p.requires_grad),
            lr=self.args.lr,
            weight_decay=1e-5,
        )

        # grad-health EMA
        self.grad_ema = {}

        # anchor snapshot for distillation
        self.anchor_net = None
        self.anchor_task = -1

    # ---------------- helpers ----------------
    def _maybe_update_anchor(self):
        cur_task = getattr(self, "current_task", 0)
        if self.anchor_net is None or self.anchor_task != cur_task:
            # snapshot current ER-ACE model as teacher
            self.anchor_net = copy.deepcopy(self.net).to(self.device)
            self.anchor_net.eval()
            for p in self.anchor_net.parameters():
                p.requires_grad = False
            self.anchor_task = cur_task

    def _build_context(self, l_new, l_buf_ce, l_buf_mse):
        eps = 1e-8
        l1 = torch.log(l_new + eps)
        l2 = torch.log(l_buf_ce + eps)
        l3 = torch.log(l_buf_mse + eps) if l_buf_mse is not None else torch.tensor(0.0, device=l_new.device)

        num_tasks = getattr(self.args, "num_tasks", 10)
        cur_task = getattr(self, "current_task", 0)
        try:
            progress_val = float(cur_task) / max(float(num_tasks) - 1.0, 1.0)
        except Exception:
            progress_val = 0.0
        progress = torch.tensor(progress_val, device=l_new.device)

        ctx = torch.stack([l1, l2, l3, progress], dim=-1)  # [4]
        return ctx.unsqueeze(0)  # [1,4]

    def _beta_grad_health_update(self, c_eff: float, rho_eff: float):
        if getattr(self.args, "dam_beta_health", 0) <= 0:
            return
        if c_eff <= 0.0 or rho_eff <= 0.0:
            return

        eta = float(getattr(self.args, "dam_beta_eta", 0.01))
        c = float(c_eff)
        rho = float(max(rho_eff, 0.0))
        eps = 1e-8

        for name, p in self.net.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.data.norm(2)
            if not torch.isfinite(g):
                continue

            if name not in self.grad_ema:
                self.grad_ema[name] = g.detach()
                continue

            ema = self.grad_ema[name]
            new_ema = (1.0 - eta) * ema + eta * g
            self.grad_ema[name] = new_ema.detach()

            r = g / (new_ema + eps)
            scale = 1.0 + c * (1.0 - r)
            min_s = 1.0 - rho
            max_s = 1.0 + rho
            scale = torch.clamp(scale, min=min_s, max=max_s)

            p.grad.data.mul_(scale)

    # ---------------- core: observe ----------------
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.net.train()
        self.controller.train()

        self.opt.zero_grad()
        self.opt_ctrl.zero_grad()

        net_params = [p for p in self.net.parameters() if p.requires_grad]
        ctrl_params = [p for p in self.controller.parameters() if p.requires_grad]

        # 1) ER-ACE forward on current minibatch
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits_all = self.net(inputs)
        mask = torch.zeros_like(logits_all)
        mask[:, present] = 1
        if self.seen_so_far.numel() > 0:
            mask[:, self.seen_so_far.max():] = 1

        logits = logits_all
        if self.current_task > 0 or self.args.task_free:
            logits = logits.masked_fill(mask == 0, -1e9)

        loss_new = self.loss(logits, labels)

        # decide whether to use replay
        use_replay = (
            len(self.buffer) > 0
            and (self.args.task_free or self.current_task > 0)
        )

        if not use_replay:
            loss_new.backward()
            grad_clip = getattr(self.args, "grad_clip", 0.0)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
            self.opt.step()

            self.buffer.add_data(examples=not_aug_inputs, labels=labels)
            return loss_new.item()

        # 2) buffer forward: CE + MSE to anchor
        buf_inputs, buf_labels = self.buffer.get_data(
            self.args.minibatch_size,
            transform=self.transform,
            device=self.device,
        )
        buf_outputs = self.net(buf_inputs)
        loss_buf_ce = self.loss(buf_outputs, buf_labels)

        # MSE distillation to anchor snapshot
        self._maybe_update_anchor()
        with torch.no_grad():
            teacher_logits = self.anchor_net(buf_inputs)
        loss_buf_mse = F.mse_loss(buf_outputs, teacher_logits)

        # 3) controller: alpha/beta eff
        with torch.no_grad():
            ctx = self._build_context(
                loss_new.detach(),
                loss_buf_ce.detach(),
                loss_buf_mse.detach(),
            )

        delta = self.controller(ctx)[0]  # [2]
        delta_alpha_raw = delta[0]       # ∈ [-1,1]
        delta_beta_raw = delta[1]        # ∈ [-1,1]

        alpha_max = float(self.args.alpha)
        gain_a = float(self.args.dam_alpha_gain)
        if alpha_max <= 0.0 or gain_a <= 0.0:
            lambda_mse = torch.tensor(0.0, device=self.device)
        else:
            # map Δα_raw ∈ [-1,1] to [0,1], then scale by alpha_max
            alpha_gate = (delta_alpha_raw * gain_a).tanh()
            alpha_gate = 0.5 * (alpha_gate + 1.0)  # ∈ [0,1]
            lambda_mse = alpha_max * alpha_gate

        beta0 = float(self.args.beta)
        gain_b = float(self.args.dam_beta_gain)
        beta_cap = float(self.args.dam_beta_cap)
        if gain_b <= 0.0:
            beta_eff = torch.tensor(beta0, device=self.device)
        else:
            beta_eff = beta0 * (1.0 + gain_b * delta_beta_raw)
            low = beta0 * (1.0 - beta_cap)
            high = beta0 * (1.0 + beta_cap)
            beta_eff = torch.clamp(beta_eff, min=low, max=high)

        # 4) anchor & total losses
        loss_anchor = loss_new + beta0 * loss_buf_ce
        loss_total = loss_new + beta_eff * loss_buf_ce + lambda_mse * loss_buf_mse

        if self.args.dam_reg > 0.0:
            reg = delta_alpha_raw ** 2 + delta_beta_raw ** 2
            loss_total = loss_total + self.args.dam_reg * reg

        # 5) gradients
        g_anchor = torch.autograd.grad(
            loss_anchor,
            net_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        g_total = torch.autograd.grad(
            loss_total,
            net_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        g_anchor_list = []
        g_total_list = []
        for p, ga, gt in zip(net_params, g_anchor, g_total):
            if ga is None:
                ga = torch.zeros_like(p)
            if gt is None:
                gt = torch.zeros_like(p)
            g_anchor_list.append(ga)
            g_total_list.append(gt)

        flat_anchor = torch.cat([ga.view(-1) for ga in g_anchor_list])
        flat_total = torch.cat([gt.view(-1) for gt in g_total_list])

        eps = 1e-8
        norm_a = flat_anchor.norm() + eps
        norm_t = flat_total.norm() + eps

        # 6) cone constraint
        if norm_a < 1e-12:
            flat_safe = flat_total.clone()
        else:
            dam_cone_cos = float(self.args.dam_cone_cos)
            dam_cone_norm_eps = float(self.args.dam_cone_norm_eps)
            if dam_cone_cos <= -1.0 and dam_cone_norm_eps >= 999.0:
                flat_safe = flat_total.clone()
            else:
                cos_val = torch.dot(flat_total, flat_anchor) / (norm_t * norm_a)

                if cos_val.item() < dam_cone_cos:
                    proj_coeff = torch.dot(flat_total, flat_anchor) / (norm_a * norm_a)
                    flat_safe = proj_coeff * flat_anchor
                else:
                    flat_safe = flat_total.clone()

                max_norm = (1.0 + dam_cone_norm_eps) * norm_a
                norm_safe = flat_safe.norm()
                if norm_safe > max_norm:
                    flat_safe = flat_safe * (max_norm / (norm_safe + eps))

        # 7) scatter safe grad back
        offset = 0
        for p in net_params:
            numel = p.numel()
            g_slice = flat_safe[offset: offset + numel].view_as(p)
            p.grad = g_slice.detach().clone()
            offset += numel

        # 8) grad-health
        num_tasks = float(getattr(self.args, "num_tasks", 10))
        cur_task = float(getattr(self, "current_task", 0))
        task_prog = cur_task / max(num_tasks - 1.0, 1.0)

        c_base = float(getattr(self.args, "dam_beta_c", 0.05))
        rho_base = float(getattr(self.args, "dam_beta_rho", 0.1))

        beta_mag = float(delta_beta_raw.abs().item())
        c_eff = c_base * beta_mag * task_prog
        rho_eff = rho_base * beta_mag * task_prog

        self._beta_grad_health_update(c_eff, rho_eff)

        # 9) controller gradients
        if ctrl_params:
            g_ctrl = torch.autograd.grad(
                loss_total,
                ctrl_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            for p, g in zip(ctrl_params, g_ctrl):
                if g is None:
                    continue
                p.grad = g.detach().clone()

        # 10) clip + step
        grad_clip = getattr(self.args, "grad_clip", 0.0)
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(), grad_clip)

        self.opt.step()
        self.opt_ctrl.step()

        # 11) update buffer
        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels,
        )

        return loss_total.item()
