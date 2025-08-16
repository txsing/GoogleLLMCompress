# Copyright 2024 DeepMind Technologies Limited (converted to PyTorch)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains a language model on the Enwik8 dataset (PyTorch version)."""

from torch.utils.data import DataLoader
from typing import Tuple
import functools
import itertools
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import math

import constants
import data_loaders

# ===========================================================
#                        数据定义
# ===========================================================

class Enwik8Dataset(Dataset):
    """Dataset for Enwik8 data."""

    def __init__(self, data_chunks) -> None:
        self.dataset = data_chunks

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.dataset[idx]
        seq_ascii = np.frombuffer(seq, dtype=np.uint8)
        # 依然回傳 uint8；模型內會轉為 long
        return torch.tensor(seq_ascii, dtype=torch.uint8)


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


# ===========================================================
#                        训练流程
# ===========================================================

# ----------------------------
# 自定义句子级 NLLoss
# ----------------------------
class SentenceLevelNLoss(nn.Module):
    def __init__(self):
        super(SentenceLevelNLoss, self).__init__()

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=-1)
        true_predictions = torch.gather(
            log_probs, 2, targets.long().unsqueeze(2)).squeeze(2)
        sentence_loss = -torch.mean(torch.sum(true_predictions, dim=1))
        return sentence_loss


class SentenceLevelNLoss(nn.Module):
    def __init__(self):
        super(SentenceLevelNLoss, self).__init__()

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=-1)
        true_predictions = torch.gather(
            log_probs, 2, targets.long().unsqueeze(2)).squeeze(2)
        sentence_loss = -torch.mean(torch.sum(true_predictions, dim=1))
        return sentence_loss


def train_transformer_decoder(
    model: nn.Module,
    data_loader: DataLoader,
    training_steps: int,
    log_every: int,
    use_tqdm: bool = True,
    device: str = 'cuda',
) -> Tuple[nn.Module, float]:

    model.to(device)
    model.train()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = SentenceLevelNLoss()

    print('Initialization done, starting training...')
    last_loss = 0.0
    data_iter = itertools.cycle(data_loader)
    for step in tqdm(range(training_steps), disable=not use_tqdm):
        batch = next(data_iter).to(device)
        optimizer.zero_grad()
        logits = model(batch)                  # [B, T, V] logits
        loss = loss_fn(logits, batch)
        loss.backward()
        optimizer.step()

        if log_every > 0 and step % log_every == 0:
            print(f'Step {step}, Loss {loss.item()}')

        last_loss = loss.item()

    return model, last_loss


def train_transformer_decoder_by_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    num_epochs: int,
    log_every: int,
    use_tqdm: bool = False,
    device: str = 'cuda',
) -> Tuple[nn.Module, float]:
    """
    按轮次训练Transformer解码器的函数。

    参数:
    model (nn.Module): 要训练的模型
    data_loader (DataLoader): 数据加载器
    num_epochs (int): 训练的轮数
    log_every (int): 每隔多少步打印一次日志
    use_tqdm (bool): 是否使用tqdm显示进度条，默认为True
    device (str): 使用的设备，默认为'cuda'

    返回:
    Tuple[nn.Module, float]: 训练后的模型和最后一轮的最后一个损失值
    """
    model.to(device)
    model.train()
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = SentenceLevelNLoss()

    print('Initialization done, starting training...')
    last_loss = 0.0
    for epoch in tqdm(range(num_epochs), disable=not use_tqdm):
        print(f"Epoch {epoch} starts!")
        for step, batch in enumerate(data_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)                  # [B, T, V] logits
            loss = loss_fn(logits, batch)
            loss.backward()
            optimizer.step()

            if log_every > 0 and step % log_every == 0:
                print(f'Epoch {epoch}, Step {step}, Loss {loss.item()}')

            last_loss = loss.item()

    return model, last_loss


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # Create config and model
    config = TransformerConfig(vocab_size=constants.ALPHABET_SIZE)
    print(config.__dict__)
    model = TransformerDecoder(config)
    print(summary(model, input_size=(
        2, constants.CHUNK_SIZE_BYTES), dtypes=[torch.long]))

    # Prepare data loader
    sequence_length = constants.CHUNK_SIZE_BYTES
    enwik8_data_generator = data_loaders.get_enwik9_iterator(
        # 只拿了 10% 用于训练，EnWik9 包含了 8，前10%是 8
        num_chunks=constants.NUM_CHUNKS // 10,  
        sequence_length=sequence_length,
    )
    enwik8_chunks = list(enwik8_data_generator)
    enwik8Dataset = Enwik8Dataset(enwik8_chunks)
    enwik8DataLoader = DataLoader(enwik8Dataset, batch_size=32, shuffle=True)

    # Start training
    model, loss = train_transformer_decoder_by_epoch(
        model=model,
        data_loader=enwik8DataLoader,
        num_epochs=2,
        log_every=500,
        device=device
    )
    print(f'Final loss: {loss}')

    # Save model
    torch.save(model.state_dict(), 'params.pth')
    print('Parameters saved in file params.pth')