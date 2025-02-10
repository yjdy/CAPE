import torch
from torch import nn

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, query,key, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        seq_len = query.shape[2] if seq_len is None else seq_len
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=query.device, dtype=query.dtype)

        cos = self.cos_cached[:seq_len].to(dtype=query.dtype)
        sin = self.sin_cached[:seq_len].to(dtype=query.dtype)
        position_ids = torch.arange(seq_len, device=query.device).unsqueeze(0)
        q_pe, k_pe = apply_rotary_pos_emb(query, key, cos, sin, position_ids)
        return q_pe, k_pe


class CAPE(nn.Module):
    def __init__ (self, dim, max_len=None, embedding_dim=None):
        '''
            CAPE：Context-aware Position Encoding
        :param dim: 每个头的通道维度
        :param max_len: 最大的上下文长度
        '''
        super().__init__()
        self.max_len = max_len if(max_len is not None) else dim
        self.pos_emb = nn.parameter.Parameter(
            torch.zeros(1, dim, self.max_len)
        )
        if embedding_dim is not None:
            self.pre_proj = nn.Sequential(nn.Linear(embedding_dim, dim),nn.SiLU())


    def forward (self, query, attention_weights) :
        '''
        :param q: (B*H,N,C) or (B,H,N,C)
        :param qk: (B*H,N,N) or (B,H,N,N), q与k相乘的结果，不包含softmax部分
        :return: E: (B*H,N,N) or (B,H,N,N)
        '''
        '''
            q(B,H,N,C) @ kT(B,H,C,N) = qk(B,H,N,N)
            q中每一个行向量与kT中的列向量相乘，因此公式表达是qiT @ kj (qi,kj都表示的列向量)
            对上下文寻址，进行门限取值,取值范围为（0,1），取值越大，意味着权重越大
            这句代码，对应了公式(3), gij = sigmid(qiT @ ki)
            G(B,H,N,N)
        '''
        G = 1 - torch.sigmoid(attention_weights)
        '''
            [B,N,N]沿着最后一个维度进行翻转特征，然后对最后一维度进行累计求和，最后再沿着最后一个维度将特征翻转刚回来
            [a,b,c] -> [c,b,a] -> [a+b+c,b+c,a] -> [a,b+c,a+b+c]
            这句代码，对应了公式(4), pij = sum{k=j ~ i}(gik)
            P(B,H,N,N)
        '''
        P = G.flip(-1).cumsum(dim=-1).flip(-1)
        P = P.clamp(max=self.max_len - 1)
        '''
            整型编码插值
            由于sigmod的原因，CAPE不能像传统的RPE一样，利用可学习的编码层学习位置信息
            因此，使用一种简单的整型向量插值方法，来融合可学习的编码特征
        '''
        P_ceil = P.ceil().long()
        P_floor = P.floor().long()
        if getattr(self, 'pre_proj', None) is not None:
            query = self.pre_proj(query)
        # (B,H,N,C) @ (1,C,T) = (B,H,N,T)
        E = torch.matmul(query, self.pos_emb) # eij
        E_ceil = E.gather(-1, P_ceil)
        E_floor = E.gather(-1, P_floor)
        P_P_floor = P - P_floor
        #E = (P - P_floor) * E_cell + (1 - P + P_floor) * E_floor
        E = P_P_floor * E_ceil + (1 - P_P_floor) * E_floor
        return E