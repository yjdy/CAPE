import torch
from torch import nn
import torch.nn.functional as F

import math

from fuxictr.pytorch.torch_utils import get_activation

class DeepseekMLP(nn.Module):
    def __init__(self, hidden_size = None, intermediate_size = None, hidden_act='silu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is None else hidden_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = get_activation(hidden_act)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class MoEGate(nn.Module):
    def __init__(self,hidden_size,
                 n_routed_experts,
                 num_experts_per_tok,
                 scoring_func='sigmoid',
                 aux_loss_alpha = 0.001,
                 seq_aux = True,
                 norm_topk_prob = True,
                 gamma=0.1,
                 routed_scaling_factor=2.5):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts

        self.scoring_func = scoring_func
        self.alpha = aux_loss_alpha
        self.seq_aux = seq_aux

        # topk selection algorithm
        self.norm_topk_prob = norm_topk_prob
        self.gating_dim = hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.biases = nn.Parameter(torch.zeros(1, self.n_routed_experts))
        self.reset_parameters()
        self.gamma = gamma
        self.routed_scaling_factor = routed_scaling_factor

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def update_biases(self, routing_weights):
        """
        动态调整偏置
        Args:
            routing_weights:

        Returns:

        """
        mask_ce = F.one_hot(routing_weights.view(-1), num_classes=self.n_routed_experts)
        ce = mask_ce.float().mean(0)
        fi = ce * self.n_routed_experts
        avg_load = fi.mean()
        var_load = fi.var()

        thred_up = avg_load + 3 * var_load
        thred_down = avg_load - 3 * var_load
        for i in range(self.n_routed_experts):
            if fi[i] > thred_up:
                self.biases[i].data -= self.gamma
            elif fi[i] < thred_down:
                self.biases[i].data += self.gamma


    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        elif self.scoring_func == 'sigmoid':
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        ### select top-k experts
        _, topk_idx = torch.topk(scores+self.biases, k=self.top_k, dim=-1, sorted=False)
        topk_weight = torch.gather(scores, -1, topk_idx)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        if self.routed_scaling_factor > 0:
            topk_weight *= self.routed_scaling_factor

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_ \
                    (seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
            self.update_biases(topk_idx_for_aux_loss)
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class DeepseekMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, hidden_size,
                 n_routed_experts=16,
                 num_experts_per_tok=6,
                 n_shared_experts=2,
                 moe_intermediate_size=None,
                 hidden_act='silu'):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.moe_intermediate_size = moe_intermediate_size if moe_intermediate_size else hidden_size
        self.experts = nn.ModuleList([DeepseekMLP(hidden_size, intermediate_size=self.moe_intermediate_size,hidden_act=hidden_act)
                                      for _ in range(self.n_routed_experts)])
        self.gate = MoEGate(hidden_size,n_routed_experts,num_experts_per_tok)
        self.n_shared_experts = n_shared_experts
        if self.n_shared_experts is not None:
            intermediate_size = moe_intermediate_size * n_shared_experts
            self.shared_experts = DeepseekMLP(hidden_size,intermediate_size=intermediate_size,hidden_act=hidden_act)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y =  y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[ i -1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache
