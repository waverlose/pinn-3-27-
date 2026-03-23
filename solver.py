"""
物理求解器模块 - 高信息报表版
包含全量物理对标与中文结构化监控输出。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad

from model import PINN
from geometry_material import GeometryMaterial
from config import *


class Solver:
    def __init__(self, lr: float = TRAIN_INIT_LR, gamma: float = TRAIN_LR_GAMMA):
        self.gm = GeometryMaterial()
        self.model = PINN().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        self.optimizer_lbfgs = torch.optim.LBFGS(self.model.parameters(), lr=1.0, history_size=20)

        self.RT = PHY_R * PHY_T 
        self.osmotic_ramp, self.chem_ramp, self.barrier_ramp = 0.0, 0.0, 0.0

        self.adaptive_weights = {
            "pde_mom": torch.tensor(1.0, device=DEVICE),
            "pde_flow": torch.tensor(1.0, device=DEVICE),
            "pde_ion": torch.tensor(1.0, device=DEVICE),
            "bc": torch.tensor(1.0, device=DEVICE),
            "trend": torch.tensor(10.0, device=DEVICE) # 初始权重
        }
        self.loss_history = {"total": [], "lr": []}
        print(f"求解器初始化完成: 三相全耦合模型已就绪")

    def set_curriculum_params(self, osmotic_ramp: float | None = None, chem_ramp: float | None = None, barrier_ramp: float | None = None):
        if osmotic_ramp is not None: self.osmotic_ramp = float(osmotic_ramp)
        if chem_ramp is not None: self.chem_ramp = float(chem_ramp)
        if barrier_ramp is not None: self.barrier_ramp = float(barrier_ramp)

    def update_adaptive_weights(self, losses_dict, last_params):
        grads = {}
        for name, loss in losses_dict.items():
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                g_list = grad(loss, last_params, retain_graph=True, allow_unused=True)
                gn = torch.sqrt(torch.sum(torch.stack([torch.norm(g)**2 for g in g_list if g is not None])) + 1e-8)
                grads[name] = gn
        if not grads: return
        base = grads.get("pde_mom", torch.tensor(1e-6, device=DEVICE))
        with torch.no_grad():
            new_weights = {}
            for name in self.adaptive_weights.keys():
                if name in grads:
                    tw = base / grads[name]
                    # 第一轮初始钳制，避免单步爆炸
                    if name == "bc": tw = torch.clamp(tw, min=1.0, max=10.0)
                    elif name == "pde_mom": tw = torch.clamp(tw, min=0.5, max=5.0)
                    elif name == "pde_ion": tw = torch.clamp(tw, min=0.3, max=5.0)
                    else: tw = torch.clamp(tw, min=0.1, max=5.0)
                    new_weights[name] = 0.9 * self.adaptive_weights[name] + 0.1 * tw
                else:
                    new_weights[name] = self.adaptive_weights[name]
            
            # 取消均值归一化，让 NTK 自由决定权重
            self.adaptive_weights["pde_mom"] = torch.clamp(new_weights.get("pde_mom", self.adaptive_weights["pde_mom"]), min=0.5, max=100.0)
            self.adaptive_weights["pde_flow"] = torch.clamp(new_weights.get("pde_flow", self.adaptive_weights["pde_flow"]), min=0.01, max=100.0)
            self.adaptive_weights["pde_ion"] = torch.clamp(new_weights.get("pde_ion", self.adaptive_weights["pde_ion"]), min=0.01, max=100.0)
            self.adaptive_weights["bc"] = torch.clamp(new_weights.get("bc", self.adaptive_weights["bc"]), min=1.0, max=100.0)
            # 趋势项权重上限调低至 50.0，确保物理主导权
            self.adaptive_weights["trend"] = torch.clamp(new_weights.get("trend", self.adaptive_weights["trend"]), min=1.0, max=50.0)

    def compute_physics(self, x_in: torch.Tensor, return_pde: bool = True, enable_chem: bool = True, retain: bool = True):
        with torch.enable_grad():
            N = x_in.shape[0]
            x = x_in.detach().clone().requires_grad_(True)
            out = self.model(x)
            ones = torch.ones((N, 1), device=DEVICE)
            # 内部求导必须 retain_graph=True，因为 out 被多次复用
            def vgrad(val): return grad(val, x, ones, create_graph=True, retain_graph=True)[0]

            # 1. 运动学
            gu, gv, gw = vgrad(out[:, 0:1]), vgrad(out[:, 1:2]), vgrad(out[:, 2:3])
            F11, F12, F13 = 1.0+gu[:,0:1], gu[:,1:2], gu[:,2:3]
            F21, F22, F23 = gv[:,0:1], 1.0+gv[:,1:2], gv[:,2:3]
            F31, F32, F33 = gw[:,0:1], gw[:,1:2], 1.0+gw[:,2:3]
            J_raw = (F11*(F22*F33-F23*F32) - F12*(F21*F33-F23*F31) + F13*(F21*F32-F22*F31))
            J = torch.abs(J_raw) + 1e-6 
            invJ = torch.clamp(1.0 / J, max=5.0)

            # 2. 材料属性
            mu_s, lam_s, phi0, fcd_ref, perm_a, perm_n, diff_a, diff_b, alpha_smooth = self.gm.get_material_params(x[:,0:1], x[:,1:2])
            ws = (1.0 - phi0) * invJ
            # 关键修正：wa (孔隙率) 必须大于一个安全下限，防止孔隙闭合导致的数值崩溃
            wa = torch.clamp(1.0 - ws, 0.05, 0.99)

            # 3. 弹性应力
            B11, B22, B33 = F11**2+F12**2+F13**2, F21**2+F22**2+F23**2, F31**2+F32**2+F33**2
            B12, B13, B23 = F11*F21+F12*F22+F13*F23, F11*F31+F12*F32+F13*F33, F21*F31+F22*F32+F23*F33
            # 弹性压力：当 J 接近 (1-phi0) 时应趋于无穷，产生极强的刚度保护
            denom = torch.clamp(J - (1.0 - phi0), min=0.005) 
            p_el = lam_s * phi0 * (J - 1.0) / (denom + 1e-7)
            tm = mu_s * invJ
            sm11, sm22, sm33 = tm*(B11-1)+p_el, tm*(B22-1)+p_el, tm*(B33-1)+p_el
            sm12, sm13, sm23 = tm*B12, tm*B13, tm*B23

            # --- 各向异性纤维强化 ---
            if ENABLE_ANISOTROPY:
                # 重新计算独立的 AF 权重：随半径增加而增强 (0 -> 1)
                r_norm_fiber = torch.sqrt((x[:,0:1]/GEO_A)**2 + (x[:,1:2]/GEO_B)**2 + 1e-12)
                t_raw = torch.clamp((r_norm_fiber - self.gm.np_ratio) / (1.0 - self.gm.np_ratio + 1e-8), 0.0, 1.0)
                t_smooth = 3 * t_raw**2 - 2 * t_raw**3
                # 修正：使用 t_smooth 使纤维在 AF 外侧最强，内侧最弱
                m_af_weight = t_smooth * (r_norm_fiber > self.gm.np_ratio).float()
                if (m_af_weight > 0.01).any():
                    beta = np.deg2rad(MAT_FIBER_THETA_INNER)
                    tx, ty = x[:,0:1], x[:,1:2]
                    r_c = torch.sqrt(tx**2 + ty**2 + 1e-9)
                    etx, ety = -ty/r_c, tx/r_c
                    cos_b, sin_b = np.cos(beta), np.sin(beta)
                    for sgn in [1.0, -1.0]:
                        a0x, a0y, a0z = sgn*cos_b*etx, sgn*cos_b*ety, sin_b
                        nfx, nfy, nfz = F11*a0x+F12*a0y+F13*a0z, F21*a0x+F22*a0y+F23*a0z, F31*a0x+F32*a0y+F33*a0z
                        lam_f2 = torch.clamp(nfx**2 + nfy**2 + nfz**2, min=1e-6)
                        EN = 0.5 * (lam_f2 - 1.0)
                        f_active = (EN > 0.0).float()
                        f_stress = MAT_FIBER_ALPHA_MAX * EN * f_active * invJ * m_af_weight
                        sm11, sm22, sm33 = sm11+f_stress*nfx**2, sm22+f_stress*nfy**2, sm33+f_stress*nfz**2
                        sm12, sm13, sm23 = sm12+f_stress*nfx*nfy, sm13+f_stress*nfx*nfz, sm23+f_stress*nfy*nfz

            # 4. 电化学 (严格对标 COMSOL)
            # 关键修正：wa0 必须是初始含水量 (phi0)，之前的 1-phi0 导致 cf 偏小 4 倍
            wa0 = phi0 
            cf = fcd_ref * wa0 / (wa * J + 1e-12)
            
            cp = 0.5 * (cf + torch.sqrt(cf**2 + 4.0 * PHY_C_EXT**2))
            c_tot = cp + (cp - cf)
            p_osmotic_current = self.RT * (c_tot - 2.0 * PHY_C_EXT)
            
            # 参考态 (J=1, wa=wa0=phi0)
            cf_ref = fcd_ref 
            cp_ref = 0.5 * (cf_ref + torch.sqrt(cf_ref**2 + 4.0 * PHY_C_EXT**2))
            c_tot_ref = cp_ref + (cp_ref - cf_ref)
            p_osmotic_ref = self.RT * (c_tot_ref - 2.0 * PHY_C_EXT)
            
            # 对标 COMSOL 的 p 变量 (Net Pore Pressure)
            # 修复: 力学孔隙压力 (out[:,3:4]) 不应受到 ramp 压制，应该只对渗透压差值进行 ramp
            p_osmotic_extra = self.osmotic_ramp * (p_osmotic_current - p_osmotic_ref)
            p_real = out[:,3:4] + p_osmotic_extra
            p_compare = out[:,3:4] + (p_osmotic_current - p_osmotic_ref) 

            
            s11, s22, s33 = sm11-p_real, sm22-p_real, sm33-p_real
            s12, s13, s23 = sm12, sm13, sm23

            if not return_pde:
                return None, None, None, None, None, (s11,s22,s33,s12,s13,s23), p_real, cf, None, wa, None, p_compare, J

            # 5. 散度与流量
            def div_X(f1, f2, f3):
                # 必须保持 retain_graph=True
                g1 = grad(f1, x, ones, create_graph=True, retain_graph=True)[0][:, 0:1]
                g2 = grad(f2, x, ones, create_graph=True, retain_graph=True)[0][:, 1:2]
                g3 = grad(f3, x, ones, create_graph=True, retain_graph=True)[0][:, 2:3]
                return torch.clamp(g1 + g2 + g3, min=-1e6, max=1e6)

            res_mx, res_my, res_mz = div_X(s11, s12, s13), div_X(s12, s22, s23), div_X(s13, s23, s33)
            
            if enable_chem:
                k_p = perm_a * torch.pow(wa/(ws+1e-8), perm_n)
                pore_r = torch.sqrt(torch.clamp(k_p, min=1e-16))
                Dp = ION_DP0 * torch.exp(-diff_a * torch.clamp(torch.pow((ION_RP/pore_r)/np.sqrt(VISCO_SI), diff_b), max=50.0))
                
                # 核心物理修正：通量由化学势梯度驱动，而非总压力梯度
                g_mu_w = vgrad(out[:,3:4]) 
                jwx, jwy, jwz = -k_p*g_mu_w[:,0:1], -k_p*g_mu_w[:,1:2], -k_p*g_mu_w[:,2:3]
                
                g_mu_i = vgrad(out[:,4:5])
                jpx, jpy, jpz = cp*jwx - wa*Dp*g_mu_i[:,0:1], cp*jwy - wa*Dp*g_mu_i[:,1:2], cp*jwz - wa*Dp*g_mu_i[:,2:3]
                
                adj11, adj12, adj13 = F22*F33-F23*F32, F13*F32-F12*F33, F12*F23-F13*F22
                adj21, adj22, adj23 = F23*F31-F21*F33, F11*F33-F13*F31, F13*F21-F11*F23
                adj31, adj32, adj33 = F21*F32-F22*F31, F12*F21-F11*F32, F11*F22-F12*F21
                res_mw = self.chem_ramp * div_X(adj11*jwx+adj12*jwy+adj13*jwz, adj21*jwx+adj22*jwy+adj23*jwz, adj31*jwx+adj32*jwy+adj33*jwz) / S_FLOW_W
                res_mi = self.chem_ramp * div_X(adj11*jpx+adj12*jpy+adj13*jpz, adj21*jpx+adj22*jpy+adj23*jpz, adj31*jpx+adj32*jpy+adj33*jpz) / S_FLOW_I
            else:
                res_mw = res_mi = torch.zeros_like(res_mx)

            # 关键保护：J 不能小于 (wsr + 0.1)，防止数值爆炸
            l_bar = torch.sum(F.relu((1.0-phi0+0.10)-J)**2) + torch.sum(F.relu(J-1.02)**2)
            return res_mx/S_MOM, res_my/S_MOM, res_mz/S_MOM, res_mw, res_mi, (s11,s22,s33,s12,s13,s23), p_real, cf, None, wa, l_bar, p_compare, J

    def train_step(self, n_dom: int, n_bc: int, target_disp_ratio: float = 0.0, pts=None, enable_chem: bool = True, update_weights: bool = False):
        self.optimizer.zero_grad()
        d_pts, b_pts, t_pts = pts["dom"].detach(), pts["btm"].detach(), pts["top"].detach()
        s_pts, s_norm = pts["side"].detach(), pts["normals"].detach()

        if update_weights:
            # 修正：采样 500 点计算 NTK 时不需要保留图
            res_w = self.compute_physics(d_pts[:500], True, enable_chem, retain=False)
            out_b_w = self.model(b_pts[:500])
            lb_m_w = torch.mean(out_b_w[:, 0:3]**2)
            # 正确解包返回值: res_mx, res_my, res_mz, res_mw, res_mi, stresses, p_real, cf, _, wa, l_bar, p_compare, J
            res_mx_w, res_my_w, res_mz_w, res_mw_w, res_mi_w, _, _, _, _, _, _, _, _ = res_w
            self.update_adaptive_weights({
                "pde_mom": torch.mean(res_mx_w**2+res_my_w**2+res_mz_w**2) + 1e-9, 
                "pde_flow": torch.mean(res_mw_w**2) + 1e-9, 
                "pde_ion": torch.mean(res_mi_w**2) + 1e-9, 
                "bc": lb_m_w + 1e-9
            }, list(self.model.output_layer.parameters()))
            
            # 关键显存清理
            del res_w, out_b_w

        res = self.compute_physics(d_pts, True, enable_chem)
        # 正确解包返回值: res_mx, res_my, res_mz, res_mw, res_mi, stresses, p_real, cf, _, wa, l_bar, p_compare, J_d
        res_mx, res_my, res_mz, res_mw, res_mi, _, p_real, cf, _, wa, l_bar, p_compare, J_d = res
        w = self.adaptive_weights
        loss_pde = w["pde_mom"]*torch.mean(res_mx**2+res_my**2+res_mz**2) + w["pde_flow"]*torch.mean(res_mw**2) + w["pde_ion"]*torch.mean(res_mi**2)
        
        # 增加 Barrier 权重
        loss_bar = (l_bar / d_pts.size(0)) * 1000.0 * self.barrier_ramp

        # --- J 拓扑形态引导 (内部均匀, 过渡区单调递增, 边界趋近1) ---
        loss_trend = torch.tensor(0.0, device=DEVICE)

        if self.barrier_ramp > 0.5:
            with torch.no_grad():
                r_norm_d = torch.sqrt((d_pts[:,0:1]/GEO_A)**2 + (d_pts[:,1:2]/GEO_B)**2).squeeze()
                m_in = r_norm_d < 0.6
                m_t1 = (r_norm_d >= 0.6) & (r_norm_d < 0.75)
                m_t2 = (r_norm_d >= 0.75) & (r_norm_d < 0.9)
                m_out = r_norm_d >= 0.9
                m_bound = r_norm_d >= 0.95

            def z_mean(mask): return J_d[mask].mean() if mask.any() else torch.tensor(1.0, device=DEVICE)

            j_in, j_t1, j_t2, j_out = z_mean(m_in), z_mean(m_t1), z_mean(m_t2), z_mean(m_out)

            # 1. 内部平均/均匀：惩罚极小化方差
            l_var = torch.var(J_d[m_in]) if m_in.any() else torch.tensor(0.0, device=DEVICE)

            # 2. 外部始终大于内部 (单调递增)
            l_mono = torch.relu(j_in - j_t1 + 1e-3) + torch.relu(j_t1 - j_t2 + 1e-3) + torch.relu(j_t2 - j_out + 1e-3)

            # 3. 边界趋近于 1
            l_bound = torch.mean((J_d[m_bound] - 1.0)**2) if m_bound.any() else torch.tensor(0.0, device=DEVICE)

            loss_trend = (l_var * 10.0 + l_mono + l_bound) * self.adaptive_weights["trend"]

        # 3. 侧面 (Side): Dirichlet mu=0, Traction-Free (补全 Z 方向剪切)
        _, _, _, _, _, s_side, _, _, _, _, _, _ = self.compute_physics(s_pts, False, enable_chem)
        nx, ny = s_norm[:, 0:1], s_norm[:, 1:2]
        # s_side: (s11, s22, s33, s12, s13, s23) -> (rr, theta, zz, r_theta, rz, theta_z)
        tx = s_side[0]*nx + s_side[3]*ny
        ty = s_side[3]*nx + s_side[1]*ny
        tz = s_side[4]*nx + s_side[5]*ny # 侧面 z 方向剪切应力
        loss_side = torch.mean(tx**2 + ty**2 + tz**2) + self.chem_ramp * torch.mean(self.model(s_pts)[:, 3:5]**2)

        # 4. 底部与顶部 (Bottom & Top): 物理平滑对标 (Dirichlet for NP/Side, Neumann for AF)
        # 权重逻辑：r_norm 从 0->1, t_smooth 从 0->1。
        # NP (r < 0.7): alpha_smooth=1 (Dirichlet), t_smooth=0 (Neumann)
        # AF (r > 0.7): alpha_smooth 从 1->0 (Neumann 渐强)
        
        # --- 底部 (Bottom) ---
        out_b = self.model(b_pts)
        lb_m_disp = torch.mean(out_b[:, 0:3]**2)
        
        r_btm = torch.sqrt((b_pts[:,0]/GEO_A)**2 + (b_pts[:,1]/GEO_B)**2 + 1e-12)
        t_af_b = torch.clamp((r_btm - NP_AXIS_RATIO) / (1.0 - NP_AXIS_RATIO + 1e-8), 0.0, 1.0)
        t_smooth_b = (3 * t_af_b**2 - 2 * t_af_b**3).unsqueeze(1)
        alpha_b = 1.0 - t_smooth_b
        
        # Dirichlet (mu=0) at NP
        lb_m_chem_d = torch.mean(alpha_b * (out_b[:, 3:4]**2 + out_b[:, 4:5]**2))
        # Neumann (no-flux) at AF
        xb = b_pts.detach().clone().requires_grad_(True)
        out_bb = self.model(xb)
        g_mu_w_b = grad(out_bb[:, 3:4], xb, torch.ones((xb.size(0), 1), device=DEVICE), create_graph=True)[0]
        g_mu_i_b = grad(out_bb[:, 4:5], xb, torch.ones((xb.size(0), 1), device=DEVICE), create_graph=True)[0]
        lb_m_chem_n = torch.mean(t_smooth_b * (g_mu_w_b[:, 2:3]**2 + g_mu_i_b[:, 2:3]**2))
        lb_m = lb_m_disp + self.chem_ramp * (lb_m_chem_d + lb_m_chem_n)

        # --- 顶部 (Top) ---
        out_t = self.model(t_pts)
        _, _, _, _, _, st, _, _, _, _, _, _ = self.compute_physics(t_pts, False, enable_chem)
        lt_m_mech = torch.mean(out_t[:, 0:2]**2) + torch.mean((st[2] + float(target_disp_ratio)*BC_TOP_PRESSURE)**2)
        
        r_top = torch.sqrt((t_pts[:,0]/GEO_A)**2 + (t_pts[:,1]/GEO_B)**2 + 1e-12)
        t_af_t = torch.clamp((r_top - NP_AXIS_RATIO) / (1.0 - NP_AXIS_RATIO + 1e-8), 0.0, 1.0)
        t_smooth_t = (3 * t_af_t**2 - 2 * t_af_t**3).unsqueeze(1)
        alpha_t = 1.0 - t_smooth_t
        
        # Dirichlet (mu=0) at NP
        lt_m_chem_d = torch.mean(alpha_t * (out_t[:, 3:4]**2 + out_t[:, 4:5]**2))
        # Neumann (no-flux) at AF
        xt = t_pts.detach().clone().requires_grad_(True)
        out_tt = self.model(xt)
        g_mu_w_t = grad(out_tt[:, 3:4], xt, torch.ones((xt.size(0), 1), device=DEVICE), create_graph=True)[0]
        g_mu_i_t = grad(out_tt[:, 4:5], xt, torch.ones((xt.size(0), 1), device=DEVICE), create_graph=True)[0]
        lt_m_chem_n = torch.mean(t_smooth_t * (g_mu_w_t[:, 2:3]**2 + g_mu_i_t[:, 2:3]**2))
        lt_m = lt_m_mech + self.chem_ramp * (lt_m_chem_d + lt_m_chem_n)

        loss_bc = w["bc"] * (lb_m + lt_m + loss_side)
        loss_total = loss_pde + loss_bc + loss_bar + loss_trend
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss_total.item(), loss_pde.item(), loss_bc.item()


    def compute_residual_norm(self, x_pts: torch.Tensor, enable_chem: bool = True):
        self.model.eval()
        all_norms = []
        batch_size = 500 # 修正：进一步减小批大小以防止 OOM

        with torch.enable_grad():
            for i in range(0, x_pts.size(0), batch_size):
                x_batch = x_pts[i:i+batch_size].detach().clone().requires_grad_(True)
                # 修正：传入 retain=False，彻底不保留评估图
                res = self.compute_physics(x_batch, True, enable_chem, retain=False)
                batch_norm = torch.sqrt(res[0]**2 + res[1]**2 + res[2]**2 + res[3]**2 + res[4]**2).detach()
                all_norms.append(batch_norm)
                # 清理显存垃圾
                del x_batch, res
                torch.cuda.empty_cache()

        self.model.train()
        return torch.cat(all_norms, dim=0)

    def debug_stats(self, x_in: torch.Tensor, epoch: int, enable_chem: bool, loss_total: float, pde_loss: float, bc_loss: float):
        with torch.no_grad():
            # 1. 计算 J 分布
            with torch.enable_grad():
                x = x_in.detach().clone().requires_grad_(True)
                out = self.model(x)
                ones = torch.ones((x.shape[0], 1), device=DEVICE)
                def vgrad(val): return grad(val, x, ones, create_graph=True, retain_graph=True)[0]
                gu, gv, gw = vgrad(out[:, 0:1]), vgrad(out[:, 1:2]), vgrad(out[:, 2:3])
                F11, F12, F13 = 1.0+gu[:,0:1], gu[:,1:2], gu[:,2:3]
                F21, F22, F23 = gv[:,0:1], 1.0+gv[:,1:2], gv[:,2:3]
                F31, F32, F33 = gw[:,0:1], gw[:,1:2], 1.0+gw[:,2:3]
                J_raw = (F11*(F22*F33-F23*F32) - F12*(F21*F33-F23*F31) + F13*(F21*F32-F22*F31))
                J_val = torch.abs(J_raw) + 1e-6

            # 2. 获取原始物理残差与压力
            res = self.compute_physics(x_in[:500], True, enable_chem)
            p_comp_val = res[11] * 1000.0 # MPa -> kPa

            # 3. 打印详细报表
            print(f"\n" + "="*70)
            print(f"[TRAIN STATS] 步数: {epoch} | 设备: {DEVICE}")
            print(f"-"*70)
            print(f"  [+] 损失统计  | 总损: {loss_total:.2e} | 物理: {pde_loss:.2e} | 边界: {bc_loss:.2e}")
            print(f"  [+] 物理残差  | 力学: {res[0].abs().mean().item():.2e} | 流量: {res[3].abs().mean().item():.2e} | 离子: {res[4].abs().mean().item():.2e}")
            print(f"  [+] 权重分配  | 动量: {self.adaptive_weights['pde_mom'].item():.2f} | 流量: {self.adaptive_weights['pde_flow'].item():.2f} | 离子: {self.adaptive_weights['pde_ion'].item():.2f} | 边界: {self.adaptive_weights['bc'].item():.2f}")
            print(f"  [+] 变形对标  | J最小: {J_val.min().item():.3f} | J最大: {J_val.max().item():.3f} | J均值: {J_val.mean().item():.3f}")
            print(f"  [+] 压力对标  | P_compare(kPa) 均值: {p_comp_val.mean().item():.1f} | 最大: {p_comp_val.max().item():.1f}")
            print(f"  [+] 课程参数  | 渗透开启: {self.osmotic_ramp*100:.1f}% | 学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
            print("="*70 + "\n")

    def train_step_lbfgs(self, n_dom, n_bc, target_disp_ratio=1.0, pts=None, enable_chem=True):
        if pts is None: return 0.0
        current_loss = [0.0]

        d_pts, b_pts, t_pts = pts["dom"].detach(), pts["btm"].detach(), pts["top"].detach()
        s_pts, s_norm = pts["side"].detach(), pts["normals"].detach()
        w = self.adaptive_weights

        def closure():
            self.optimizer_lbfgs.zero_grad()

            # 1. 域内 PDE 与 Barrier (核心物理) - 修正解包接收 13 个值
            mx, my, mz, mw, mi, _, _, _, _, _, lb, _, J_d = self.compute_physics(d_pts, True, enable_chem)
            loss_pde = w["pde_mom"]*torch.mean(mx**2+my**2+mz**2) + w["pde_flow"]*torch.mean(mw**2) + w["pde_ion"]*torch.mean(mi**2)
            loss_bar = (lb / d_pts.size(0)) * 5000.0 * self.barrier_ramp

            # --- J 拓扑形态引导 (内部均匀, 过渡区单调递增, 边界趋近1) ---
            loss_trend = torch.tensor(0.0, device=DEVICE)
            if self.barrier_ramp > 0.5:
                with torch.no_grad():
                    rn = torch.sqrt((d_pts[:,0:1]/GEO_A)**2 + (d_pts[:,1:2]/GEO_B)**2).squeeze()
                    m_in = rn < 0.6
                    m_t1 = (rn >= 0.6) & (rn < 0.75)
                    m_t2 = (rn >= 0.75) & (rn < 0.9)
                    m_out = rn >= 0.9
                    m_bound = rn >= 0.95
                    
                def z_mean(mask): return J_d[mask].mean() if mask.any() else torch.tensor(1.0, device=DEVICE)
                
                j_in, j_t1, j_t2, j_out = z_mean(m_in), z_mean(m_t1), z_mean(m_t2), z_mean(m_out)
                
                l_var = torch.var(J_d[m_in]) if m_in.any() else torch.tensor(0.0, device=DEVICE)
                l_mono = torch.relu(j_in - j_t1 + 1e-3) + torch.relu(j_t1 - j_t2 + 1e-3) + torch.relu(j_t2 - j_out + 1e-3)
                l_bound = torch.mean((J_d[m_bound] - 1.0)**2) if m_bound.any() else torch.tensor(0.0, device=DEVICE)
                
                loss_trend = (l_var * 10.0 + l_mono + l_bound) * w["trend"]

            # 2. 侧边边界 (Side BC): Dirichlet mu=0, Traction-Free
            _, _, _, _, _, s_side, _, _, _, _, _, _ = self.compute_physics(s_pts, False, enable_chem)
            nx, ny = s_norm[:, 0:1], s_norm[:, 1:2]
            tx = s_side[0]*nx + s_side[3]*ny
            ty = s_side[3]*nx + s_side[1]*ny
            loss_side_mech = torch.mean(tx**2 + ty**2)
            out_s = self.model(s_pts)
            loss_side_chem = self.chem_ramp * 10.0 * torch.mean(out_s[:, 3:5]**2)
            loss_side = loss_side_mech + loss_side_chem

            # 3. 底部与顶部边界 (Bottom & Top): 平滑对标
            # --- 底部 ---
            out_b = self.model(b_pts)
            lb_m_disp = torch.mean(out_b[:, 0:3]**2)
            r_btm = torch.sqrt((b_pts[:,0]/GEO_A)**2 + (b_pts[:,1]/GEO_B)**2 + 1e-12)
            t_af_b = torch.clamp((r_btm - NP_AXIS_RATIO) / (1.0 - NP_AXIS_RATIO + 1e-8), 0.0, 1.0)
            t_smooth_b = (3 * t_af_b**2 - 2 * t_af_b**3).unsqueeze(1)
            alpha_b = 1.0 - t_smooth_b
            lb_m_chem_d = torch.mean(alpha_b * (out_b[:, 3:5]**2))
            
            xb = b_pts.detach().clone().requires_grad_(True)
            out_bb = self.model(xb)
            g_mu_b = grad(out_bb[:, 3:5], xb, torch.ones((xb.size(0), 2), device=DEVICE), create_graph=True)[0]
            lb_m_chem_n = torch.mean(t_smooth_b * (g_mu_b[:, 2:3]**2)) # 仅对轴向梯度做 Neumann
            lb_m = lb_m_disp + self.chem_ramp * (lb_m_chem_d + lb_m_chem_n)

            # --- 顶部 ---
            out_t = self.model(t_pts)
            _, _, _, _, _, st, _, _, _, _, _, _ = self.compute_physics(t_pts, False, enable_chem)
            lt_m_mech = torch.mean(out_t[:, 0:2]**2) + torch.mean((st[2] + float(target_disp_ratio)*BC_TOP_PRESSURE)**2)
            
            r_top = torch.sqrt((t_pts[:,0]/GEO_A)**2 + (t_pts[:,1]/GEO_B)**2 + 1e-12)
            t_af_t = torch.clamp((r_top - NP_AXIS_RATIO) / (1.0 - NP_AXIS_RATIO + 1e-8), 0.0, 1.0)
            t_smooth_t = (3 * t_af_t**2 - 2 * t_af_t**3).unsqueeze(1)
            alpha_t = 1.0 - t_smooth_t
            lt_m_chem_d = torch.mean(alpha_t * (out_t[:, 3:5]**2))
            
            xt = t_pts.detach().clone().requires_grad_(True)
            out_tt = self.model(xt)
            g_mu_t = grad(out_tt[:, 3:5], xt, torch.ones((xt.size(0), 2), device=DEVICE), create_graph=True)[0]
            lt_m_chem_n = torch.mean(t_smooth_t * (g_mu_t[:, 2:3]**2))
            lt_m = lt_m_mech + self.chem_ramp * (lt_m_chem_d + lt_m_chem_n)

            # 4. 总损失
            loss_bc = w["bc"] * (lb_m + lt_m + loss_side)
            loss = loss_pde + loss_bc + loss_bar + loss_trend + loss_smooth

            loss.backward()
            current_loss[0] = loss.item()
            return loss

        self.optimizer_lbfgs.step(closure)
        return current_loss[0]