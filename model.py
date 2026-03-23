"""
PINN神经网络模型模块 (SIREN 经典单网络版)
Physics-Informed Neural Network Model Module (Standard SIREN)
"""
import torch
import torch.nn as nn
import numpy as np
from config import *

class SirenLayer(nn.Module):
    """
    自适应 SIREN 层 (Adaptive SIREN Layer)
    保留可学习频率因子，支持单网络架构。
    """
    def __init__(self, in_features, out_features, w0=1.0, is_first=False):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.adaptive_w = nn.Parameter(torch.ones(1) * w0)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                n = self.linear.in_features
                self.linear.weight.uniform_(-1 / n, 1 / n)
            else:
                n = self.linear.in_features
                limit = np.sqrt(6 / n) / self.w0
                self.linear.weight.uniform_(-limit, limit)
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        return torch.sin(self.adaptive_w * self.linear(x))

class PINN(nn.Module):
    """
    物理信息神经网络 (SIREN 单网络版)
    回归单 MLP 架构以减少计算量和显存压力。
    """
    def __init__(self, hidden_dim=NET_HIDDEN_DIM, num_layers=NET_NUM_LAYERS):
        super().__init__()
        
        layers = []
        # 输入层
        layers.append(SirenLayer(3, hidden_dim, w0=1.0, is_first=True))
        # 隐藏层
        for _ in range(num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, w0=1.0, is_first=False))
        
        self.net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, 5) # 输出 [u, v, w, mu_w, mu_ion]
        self.out_layer = self.output_layer # 兼容别名
        
        # 初始化输出层
        with torch.no_grad():
            self.output_layer.weight.fill_(0.0)
            self.output_layer.bias.fill_(0.0)
            self.output_layer.weight.uniform_(-1e-4, 1e-4)
        
        # 预注册归一化常数
        self.register_buffer('inv_geo', torch.tensor([1.0/GEO_A, 1.0/GEO_B, 1.0/GEO_H], dtype=torch.float32))
    
    def forward(self, x_in):
        x, y, z = x_in[:, 0:1], x_in[:, 1:2], x_in[:, 2:3]
        feats = x_in * self.inv_geo
        
        raw_out = self.output_layer(self.net(feats))
        
        # 物理映射 (Hard Constraints & Scaling)
        hat_u = raw_out[:, 0:1] * MODEL_OUT_SCALE_U
        hat_v = raw_out[:, 1:2] * MODEL_OUT_SCALE_U
        w     = raw_out[:, 2:3] * MODEL_OUT_SCALE_U
        
        # 椭圆对称性硬约束
        u = x * hat_u
        v = y * hat_v
        
        mu_w   = raw_out[:, 3:4] * MODEL_OUT_SCALE_MU
        mu_ion = raw_out[:, 4:5] * MODEL_OUT_SCALE_MU
        
        return torch.cat([u, v, w, mu_w, mu_ion], dim=1)
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=DEVICE))
