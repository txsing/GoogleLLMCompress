import torch
import torch.nn as nn
import math
import constants

# ===========================================================
#                        模型定义
# ===========================================================
class TransformerConfig:
    """Hyperparameters used in the Transformer architectures."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        emb_init_scale: float = 0.02,
        widening_factor: int = 4,
        dropout: float = 0.0,
        max_length: int = constants.CHUNK_SIZE_BYTES,
        bos_token_id: int = 0,
        tie_weights: bool = False,  # 如需與輸入 embedding 綁定，改成 True
    ) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.emb_init_scale = emb_init_scale
        self.widening_factor = widening_factor
        self.dropout = dropout
        self.max_length = max_length
        self.bos_token_id = bos_token_id
        self.tie_weights = tie_weights


# ----------------------------
# Sinusoidal Positional Encoding
# ----------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(
            1)                # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))
        # [T, D]
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> 回傳 [T, D]，之後會 broadcast 到 batch 維度
        T = x.size(1)
        return self.pe[:T]


# ----------------------------
# 使用 PyTorch 內建 TransformerDecoder 的語言模型
# ----------------------------

class TransformerDecoder(nn.Module):
    """Transformer decoder model (PyTorch built-ins)."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
        )
        # 对神经网络中的 embedding layer 权重进行初始化
        # 使用的是截断正态分布（truncated normal distribution）
        nn.init.trunc_normal_(self.embedding.weight, std=config.emb_init_scale)
        # Positional encoding（固定正弦）
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=config.embedding_dim,
            max_len=config.max_length,
        )
        # 內建 Decoder Layer + 堆疊
        d_model = config.embedding_dim
        dim_ff = d_model * config.widening_factor
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=config.num_heads,
            dim_feedforward=dim_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,      # 讓張量是 [B, T, D]
            norm_first=True,       # Pre-LN 比較穩定
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=config.num_layers,
        )
        # 最終 LayerNorm（常見做法，可留可去）
        self.final_norm = nn.LayerNorm(d_model)
        # 輸出線性層
        self.output_layer = nn.Linear(d_model, config.vocab_size, bias=False)
        # 可選：權重綁定（weight tying）
        # 是否让 output 层的權重與 input embedding 層的權重綁定（一般默认是 fasle）
        if config.tie_weights:
            self.output_layer.weight = self.embedding.weight

    @torch.no_grad()
    def _causal_mask(self, T: int, device) -> torch.Tensor:
        # PyTorch 的 attn_mask: True 表示不允許注意（被遮蔽）
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def shift_right(self, sequences: torch.Tensor) -> torch.Tensor:
        """Right-shift the input by padding on the temporal axis."""
        sequences = sequences.long()
        bos = torch.full(
            (sequences.size(0), 1),
            fill_value=self.config.bos_token_id,
            dtype=sequences.dtype,
            device=sequences.device,
        )
        return torch.cat([bos, sequences[:, :-1]], dim=1)

    def forward(self, targets: torch.Tensor) -> torch.Tensor:

        # 右移得到自迴歸輸入，比如原始输入是 abc ，实际模型接收<bos>ab
        inputs = self.shift_right(targets)                   # [B, T]

        # Token + Positional
        x = self.embedding(inputs)                           # [B, T, D]
        x = x * math.sqrt(self.config.embedding_dim)
        pos = self.pos_encoding(x).to(x.device)              # [T, D]
        # broadcast 到 [B, T, D]
        x = x + pos

        # 因果遮罩（純 LM 無 encoder memory）
        T = x.size(1)
        attn_mask = self._causal_mask(T, x.device)           # [T, T] bool

        # Decoder forward（不需要 memory）
        h = self.decoder(tgt=x, memory=x, tgt_mask=attn_mask)
        h = self.final_norm(h)

        logits = self.output_layer(h)                        # [B, T, V]
        return logits