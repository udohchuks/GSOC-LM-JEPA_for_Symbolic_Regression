"""
Transformer for LM-JEPA symbolic regression.
Architecture inspired by LLaMA/Qwen:
  - Token embedding (shared between input and LM head)
  - Rotary positional encoding
  - N transformer layers (standard pre-norm)

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


CONFIG = {
    "emb_dim":  256,
    "hidden_dim": 1024,
    "num_heads": 8,
    "head_dim": 32,      # emb_dim // num_heads = 512 // 8
    "context_length": 512,
    "vocab_size": 110, # 46 equation + 64 bin tokens
    "n_layers": 6,
    "dtype": torch.bfloat16,
    "len_a": 220, # max View A length (9-var equation)
    "len_b": 40, # max View B length
    "pad_id": 0,
    "dropout": 0.1,
}


class ROPE(nn.Module):
    def __init__(self, head_dim, context_length,  dtype=torch.float32):
        super().__init__()
        theta_base = 10000
        self.head_dim = head_dim
        self.context_length = context_length

        assert self.head_dim % 2 == 0, "head_dim must be even"

        theta = 1.0 / (theta_base ** (torch.arange(0, self.head_dim, 2, dtype=dtype)[: (self.head_dim // 2)].float() / self.head_dim))

        self.register_buffer('theta', theta)
        self._build_cache()

    def _build_cache(self):
        positions = torch.arange(self.context_length).float()
        angles = torch.outer(positions, self.theta) # shape -> (context_length, head_dim//2)

        self.register_buffer('cos_cache', angles.cos(), persistent=False)
        self.register_buffer('sin_cache', angles.sin(), persistent=False)
        
    
    def forward(self, x):
        _, _, seq_len, h_dim = x.shape

        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]

        cos = self.cos_cache[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len, :].unsqueeze(0).unsqueeze(0)

        x_rot_1 = x1 * cos - x2 * sin
        x_rot_2 = x1 * sin + x2 * cos

        return torch.cat([x_rot_1, x_rot_2], dim=-1).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f* torch.rsqrt(var + self.eps)
        
        out = x_norm * (1 + self.scale)

        return out.to(input_dtype)
    

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg['emb_dim'], cfg['hidden_dim'], dtype=cfg['dtype'], bias=False)
        self.up_proj = nn.Linear(cfg['emb_dim'], cfg['hidden_dim'], dtype=cfg['dtype'], bias=False)
        self.down_proj = nn.Linear(cfg['hidden_dim'], cfg['emb_dim'], dtype=cfg['dtype'], bias=False)
    
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)

        return self.down_proj(gate * up)

class MLA(nn.Module):
    def __init__(self, emb_dim, num_heads, head_dim, context_length, dropout=0.1, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.d_out     = num_heads * head_dim

        self.W_queries = nn.Linear(emb_dim, self.d_out, bias=False, dtype=dtype)
        self.W_keys = nn.Linear(emb_dim, self.d_out, bias=False, dtype=dtype)
        self.W_values = nn.Linear(emb_dim, self.d_out, bias=False, dtype=dtype)

        self.W_o = nn.Linear(self.d_out, emb_dim, bias=False, dtype=dtype)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        self.apply_rope = ROPE(head_dim, context_length)

        self.scaling = (head_dim) ** -0.5

    def forward(self, x, mask):
        b, num_tokens, _ = x.shape

        queries = self.W_queries(x).view(b, num_tokens, self.num_heads, self.head_dim)
        keys = self.W_keys(x).view(b, num_tokens, self.num_heads, self.head_dim)
        values = self.W_values(x).view(b, num_tokens, self.num_heads, self.head_dim)

        queries = self.q_norm(queries).transpose(1, 2)
        keys = self.k_norm(keys).transpose(1, 2)
        values = values.transpose(1, 2)

        keys = self.apply_rope(keys)
        queries = self.apply_rope(queries)

        queries = queries * self.scaling

        attn_scores = queries @ keys.transpose(2, 3)
        if mask is not None:
            # mask: [S, S] bool — True=block, False=allow
            attn_scores = attn_scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(0), float('-inf')
            )

        attn_weights = torch.softmax(attn_scores, dim=-1)

        out = (attn_weights @ values).transpose(1, 2).contiguous().reshape(b, num_tokens, self.d_out)

        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.mla = MLA(
            emb_dim = cfg['emb_dim'],
            num_heads = cfg['num_heads'],
            head_dim = cfg['head_dim'],
            context_length = cfg['context_length'],
            dropout = cfg['dropout'],
            dtype = cfg['dtype'],
        )
        self.ff = FeedForward(cfg)
        self.input_layer_norm = RMSNorm(cfg['emb_dim'])
        self.post_attn_norm = RMSNorm(cfg['emb_dim'])
        self.pre_ffn_norm = RMSNorm(cfg['emb_dim'])
        self.post_ffn_norm = RMSNorm(cfg['emb_dim'])

    def forward(
        self,
        x,
        mask,
    ):
        shortcut = x
        x = self.input_layer_norm(x)
        x_attn = self.mla(x, mask)
        x_attn = self.post_attn_norm(x_attn)
        x = x_attn + shortcut

        shortcut = x
        x_fnn = self.pre_ffn_norm(x)
        x_ffn = self.ff(x)
        x_ffn = self.post_ffn_norm(x_ffn)

        x = x_ffn + shortcut

        return x


class CustomTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.token_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'], dtype=cfg['dtype'])
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg['n_layers'])
        ])
        self.final_norm = RMSNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False, dtype=cfg['dtype'])
        self.cfg = cfg
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def get_lm_logits(self, hidden):
        logits = self.out_head(hidden)

        return logits
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)
    
    def forward(self, token_ids, mask):
        """
        token_ids: [B, seq_len]
        mask:      [seq_len, seq_len] bool — True=block
        returns:   [B, seq_len, emb_dim]
        """
        b, seq_len = token_ids.shape
        x = self.token_emb(token_ids)

        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        return x # [B, seq_len, d_model]
        

def build_causal_mask(seq_len, device=None):
    if device is None: 
        device = torch.device('cpu')
    ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    mask_global = torch.triu(ones, diagonal=1)

    return mask_global

def build_jepa_mask(len_a, len_b, device=None):
    """
    JEPA mask for layout [View A | View B | PRED].

    View A: causal within itself
    View B: sees all View A + causal within itself
    PRED:   sees all View A only — blind to View B
    """
    total = len_a + len_b + 1
    mask = torch.ones(total, total, dtype=torch.bool, device=device)

    for i in range(len_a):
        mask[i, :i + 1] = False
    for i in range(len_b):
        abs_i = len_a + i

        mask[abs_i, :len_a] = False
        mask[abs_i, len_a:abs_i + 1] = False
    pred_pos = len_a + len_b

    mask[pred_pos, :len_a] = False

    return mask

