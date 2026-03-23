"""
几何和材料属性模块
Geometry and Material Properties Module - 统一参数引用
"""
import torch
import numpy as np
from config import *


class GeometryMaterial:
    def __init__(self):
        self.a = GEO_A
        self.b = GEO_B
        self.h = GEO_H
        self.np_ratio = NP_AXIS_RATIO

    def sample_domain(self, n):
        # 稳健采样：循环直到凑够 n 个点
        points = torch.empty((0, 3), device=DEVICE)
        while len(points) < n:
            n_req = n - len(points)
            n_sample = int(n_req * 1.5) + 100 # 多采 50% + buffer
            
            x = (torch.rand(n_sample, device=DEVICE) * 2 - 1) * self.a
            y = (torch.rand(n_sample, device=DEVICE) * 2 - 1) * self.b
            z = torch.rand(n_sample, device=DEVICE) * self.h
            
            mask = (x / self.a) ** 2 + (y / self.b) ** 2 <= 1.0
            new_pts = torch.stack([x[mask], y[mask], z[mask]], dim=1)
            points = torch.cat([points, new_pts], dim=0)
            
        return points[:n]

    def sample_boundary_top_bottom(self, n):
        n_half = n // 2
        theta = torch.rand(n_half, device=DEVICE) * 2 * np.pi
        r = torch.sqrt(torch.rand(n_half, device=DEVICE))
        x_btm, y_btm = r * self.a * torch.cos(theta), r * self.b * torch.sin(theta)
        z_btm = torch.zeros_like(x_btm)
        theta2 = torch.rand(n_half, device=DEVICE) * 2 * np.pi
        r2 = torch.sqrt(torch.rand(n_half, device=DEVICE))
        x_top, y_top = r2 * self.a * torch.cos(theta2), r2 * self.b * torch.sin(theta2)
        z_top = torch.ones_like(x_top) * self.h
        return torch.stack([x_btm, y_btm, z_btm], dim=1), torch.stack([x_top, y_top, z_top], dim=1)

    def sample_boundary_side(self, n):
        theta = torch.rand(n, device=DEVICE) * 2 * np.pi
        z = torch.rand(n, device=DEVICE) * self.h
        x, y = self.a * torch.cos(theta), self.b * torch.sin(theta)
        nx = 2 * x / (self.a ** 2)
        ny = 2 * y / (self.b ** 2)
        norm = torch.sqrt(nx**2 + ny**2 + 1e-12)
        nx, ny = nx / norm, ny / norm
        return torch.stack([x, y, z], dim=1), torch.stack([nx, ny, torch.zeros_like(nx)], dim=1)

    def _compute_alpha(self, x, y):
        # r goes from 0 to 1
        r = torch.sqrt((x / self.a) ** 2 + (y / self.b) ** 2 + 1e-6)
        t = (r - self.np_ratio) / (1.0 - self.np_ratio + 1e-8)
        t = torch.clamp(t, 0.0, 1.0)
        # alpha=1 at center (NP), alpha=0 at boundary (AF)
        alpha = 1.0 - (3 * t**2 - 2 * t**3)
        return alpha

    def get_material_params(self, x, y):
        # r 归一化半径 (0 to 1)
        r_norm = torch.sqrt((x / self.a) ** 2 + (y / self.b) ** 2 + 1e-12)
        
        # 计算 AF 内部的线性插值因子 (NP 边界处为 1, 外部边界处为 0)
        t_af = (r_norm - self.np_ratio) / (1.0 - self.np_ratio + 1e-8)
        t_af = torch.clamp(t_af, 0.0, 1.0)
        alpha_af = 1.0 - t_af # 线性下降
        
        # 区分 NP 和 AF
        is_np = (r_norm < self.np_ratio)
        
        # --- 对标 comsolsettings.csv 的 GAG 分布 (C2 平滑五次 Hermite 版本) ---
        # 五次 Hermite 保证端点处的一阶、二阶导数均为 0
        alpha_smooth = 1.0 - (6 * t_af**5 - 15 * t_af**4 + 10 * t_af**3)
        
        # 使用 alpha_smooth 替换 alpha_af 确保连续性
        fcd_np = torch.full_like(r_norm, 3.4e-7)
        fcd_af = (3.4e-7 * alpha_smooth + 1.6e-7 * (1.0 - alpha_smooth))
        fcd = torch.where(is_np, fcd_np, fcd_af)
        
        # 弹性参数与物理分率插值
        mu = alpha_smooth * MAT_NP_MU + (1 - alpha_smooth) * MAT_AF_MU
        lam = alpha_smooth * MAT_NP_LAMBDA + (1 - alpha_smooth) * MAT_AF_LAMBDA
        wsr = alpha_smooth * MAT_NP_WSR + (1 - alpha_smooth) * MAT_AF_WSR
        phi0 = 1.0 - wsr
        
        # 渗透率与扩散参数
        perm_a = alpha_smooth * MAT_NP_PERM_A + (1 - alpha_smooth) * MAT_AF_PERM_A
        perm_n = alpha_smooth * MAT_NP_PERM_N + (1 - alpha_smooth) * MAT_AF_PERM_N
        diff_a = alpha_smooth * MAT_NP_DIFF_A + (1 - alpha_smooth) * MAT_AF_DIFF_A
        diff_b = alpha_smooth * MAT_NP_DIFF_B + (1 - alpha_smooth) * MAT_AF_DIFF_B
        
        return mu, lam, phi0, fcd, perm_a, perm_n, diff_a, diff_b, alpha_smooth

    def get_porosity_from_wsr(self, wsr):
        return 1.0 - wsr

    def get_permeability_custom(self, phi, ws, perm_a, perm_n):
        return perm_a * torch.pow(phi / (ws + 1e-8), perm_n)