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
import itertools
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import constants
import data_loaders

import argparse
from transformer_torch import TransformerConfig, TransformerDecoder

parser = argparse.ArgumentParser(description='Train Transformer Decoder on Enwik8 dataset.')
parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--drill', action='store_true', default=False, help='drill run')

args = parser.parse_args()


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
            if args.drill and step > 10:
                break
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
    batch_size = 8 if args.drill else args.batch_size
    enwik8DataLoader = DataLoader(enwik8Dataset, batch_size=batch_size, shuffle=True)

    # Start training
    model, loss = train_transformer_decoder_by_epoch(
        model=model,
        data_loader=enwik8DataLoader,
        num_epochs=args.epochs,
        log_every=500,
        device=device
    )
    print(f'Final loss: {loss}')

    # Save model
model_filename = 'params_drill.pth' if args.drill else f'params_epoch_{args.epochs}.pth'
torch.save(model.state_dict(), model_filename)
print(f'Parameters saved in file {model_filename}')