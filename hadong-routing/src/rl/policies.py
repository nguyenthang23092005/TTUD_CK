"""Bộ trích đặc trưng GNN trên cây lân cận (mục B1) cho Stable-Baselines3.

Quan sát (chế độ `tree`): [đặc trưng toàn cục (G)] + [khe tầng 1 (K)] + [tầng 2 (K²)] + … + [tầng D (K^D)], mỗi khe
F đặc trưng (cột 0 = cờ hợp lệ). Truyền thông điệp từ dưới lên (con → cha) với trọng số dùng chung giữa các khe
cùng tầng:
    m_D      = φ_D(x_D)                           (mã hoá khe lá)
    m_d      = φ_d([x_d, max_{con c} m_{d+1}(c)])  (gộp max các con hợp lệ)
Đầu ra = [ψ(toàn cục), m_1(nhánh 1), …, m_1(nhánh K)] — mỗi nhánh tầng 1 ứng với một hành động, nên chính sách có
thể đánh giá "phía sau" mỗi cạnh ra tới D bước (đặc biệt: phát hiện ngõ cụt). Đây là GNN trên cây BFS của đồ thị cục bộ.
"""
from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class TreeGNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, n_global: int, K: int, depth: int, n_feat: int,
                 hidden: int = 64) -> None:
        super().__init__(observation_space, features_dim=hidden + K * hidden)
        self.G, self.K, self.D, self.F, self.H = n_global, K, depth, n_feat, hidden
        self.glob = nn.Sequential(nn.Linear(n_global, hidden), nn.ReLU())
        self.enc = nn.ModuleList()
        for d in range(depth, 0, -1):
            n_in = n_feat + (hidden if d < depth else 0)
            self.enc.append(nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()))
        sizes = [K ** d for d in range(1, depth + 1)]
        self.offsets = [n_global + sum(sizes[:i]) * n_feat for i in range(depth)]
        self.sizes = sizes

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        child: torch.Tensor | None = None
        for i, d in enumerate(range(self.D, 0, -1)):
            n = self.sizes[d - 1]
            x = obs[:, self.offsets[d - 1]: self.offsets[d - 1] + n * self.F].reshape(B, n, self.F)
            valid = x[..., :1]
            inp = x if child is None else torch.cat([x, child], dim=-1)
            m = self.enc[i](inp) * valid
            if d > 1:
                # gộp K con của mỗi khe cha: (B, n/K, K, H) -> max
                m = m.reshape(B, n // self.K, self.K, self.H)
                child = torch.where(valid.reshape(B, n // self.K, self.K, 1) > 0.5, m,
                                    torch.full_like(m, -1e4)).max(dim=2).values
                child = torch.where(child < -1e3, torch.zeros_like(child), child)
            else:
                level1 = m
        return torch.cat([self.glob(obs[:, : self.G]), level1.reshape(B, -1)], dim=1)
