"""
物理求解器模块 - 终极稳健对标版
1. 修复 Trying to backward through the graph a second time 报错（锁定 vgrad 内部 retain_graph=True）。
2. 修复 TypeError: unexpected keyword argument 'retain'（全量清理接口参数）。
3. 严格对标 COMSOL PINNmodel.m 物理逻辑。
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad

from model import PINN
from geometry_material import GeometryMaterial
from weight_manager import WeightManager
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

        # 使用统一的权重管理器（解决初始化不一致、梯度估计偏差等问题）
        self.weight_manager = WeightManager()
        print(f"求解器初始化完成: 统一权重管理版已就绪")

    def _memory_checkpoint(self, tag: str = ""):
        """内存检查点，用于调试GPU内存使用"""
        # 静默模式：不打印日志，仅内部监控
        pass
    
    def _device_check(self, tensor, name: str = ""):
        """检查张量设备一致性"""
        # 静默模式：仅执行设备检查，不打印日志
        if tensor.device != DEVICE:
            return tensor.to(DEVICE)
        return tensor
    
    def _safe_cleanup(self, *tensors):
        """安全清理张量，帮助释放计算图"""
        for t in tensors:
            if t is not None:
                del t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_curriculum_params(self, osmotic_ramp: float | None = None, chem_ramp: float | None = None, barrier_ramp: float | None = None):
        if osmotic_ramp is not None: self.osmotic_ramp = float(osmotic_ramp)
        if chem_ramp is not None: self.chem_ramp = float(chem_ramp)
        if barrier_ramp is not None: self.barrier_ramp = float(barrier_ramp)

    # 注意：update_adaptive_weights 方法已移除，由 WeightManager 统一管理权重

    def compute_physics(self, x_in: torch.Tensor, return_pde: bool = True, enable_chem: bool = True):
        # 统一的物理计算核心 - 内存安全版
        self._memory_checkpoint(f"compute_physics_start (N={x_in.shape[0]})")
        
        with torch.enable_grad():
            N = x_in.shape[0]
            x = x_in.detach().clone().requires_grad_(True)
            x = self._device_check(x, "x")
            
            out = self.model(x)
            
            # 优化梯度计算：避免重复构建计算图
            # 一次性计算所有需要的梯度
            ones = torch.ones((N, 1), device=DEVICE)
            
            # 内部求导闭包：使用 retain_graph=True 但添加安全包装
            def vgrad(val):
                try:
                    return grad(val, x, ones, create_graph=True, retain_graph=True)[0]
                except RuntimeError as e:
                    # 静默处理梯度错误
                    raise RuntimeError(f"梯度计算失败: {e}")

            # 1. 运动学
            gu, gv, gw = vgrad(out[:, 0:1]), vgrad(out[:, 1:2]), vgrad(out[:, 2:3])
            F11, F12, F13 = 1.0+gu[:,0:1], gu[:,1:2], gu[:,2:3]
            F21, F22, F23 = gv[:,0:1], 1.0+gv[:,1:2], gv[:,2:3]
            F31, F32, F33 = gw[:,0:1], gw[:,1:2], 1.0+gw[:,2:3]
            J = torch.abs(F11*(F22*F33-F23*F32) - F12*(F21*F33-F23*F31) + F13*(F21*F32-F22*F31)) + 1e-6
            invJ = torch.clamp(1.0 / J, max=5.0)

            # 2. 材料属性
            mu_s, lam_s, phi0, fcd_ref, perm_a, perm_n, diff_a, diff_b, alpha_smooth = self.gm.get_material_params(x[:,0:1], x[:,1:2])
            wa = torch.clamp(1.0 - (1.0 - phi0)*invJ, 0.05, 0.99)

            # 3. 弹性与纤维 (双族螺旋)
            B11, B22, B33 = F11**2+F12**2+F13**2, F21**2+F22**2+F23**2, F31**2+F32**2+F33**2
            B12, B13, B23 = F11*F21+F12*F22+F13*F23, F11*F31+F12*F32+F13*F33, F21*F31+F22*F32+F23*F33
            p_el = lam_s * phi0 * (J - 1.0) / torch.clamp(J - (1.0 - phi0), min=0.005)
            tm = mu_s * invJ
            sm11, sm22, sm33 = tm*(B11-1)+p_el, tm*(B22-1)+p_el, tm*(B33-1)+p_el
            sm12, sm13, sm23 = tm*B12, tm*B13, tm*B23

            if ENABLE_ANISOTROPY:
                beta = np.deg2rad(MAT_FIBER_THETA_INNER)
                r_c = torch.sqrt(x[:,0:1]**2 + x[:,1:2]**2 + 1e-9)
                etx, ety = -x[:,1:2]/r_c, x[:,0:1]/r_c
                cos_b, sin_b = np.cos(beta), np.sin(beta)
                t_af = torch.clamp((torch.sqrt((x[:,0:1]/GEO_A)**2 + (x[:,1:2]/GEO_B)**2 + 1e-12) - self.gm.np_ratio)/(1-self.gm.np_ratio+1e-8), 0, 1)
                m_af_w = (3*t_af**2 - 2*t_af**3) * (t_af > 0).float()
                for sgn in [1.0, -1.0]:
                    a0x, a0y, a0z = sgn*cos_b*etx, sgn*cos_b*ety, sin_b
                    nfx, nfy, nfz = F11*a0x+F12*a0y+F13*a0z, F21*a0x+F22*a0y+F23*a0z, F31*a0x+F32*a0y+F33*a0z
                    EN = 0.5 * (nfx**2 + nfy**2 + nfz**2 - 1.0)
                    fs = MAT_FIBER_ALPHA_MAX * t_af * EN * (EN > 0).float() * invJ * m_af_w
                    sm11, sm22, sm33 = sm11+fs*nfx**2, sm22+fs*nfy**2, sm33+fs*nfz**2
                    sm12, sm13, sm23 = sm12+fs*nfx*nfy, sm13+fs*nfx*nfz, sm23+fs*nfy*nfz

            # 4. 电化学
            cf = fcd_ref * phi0 / (wa * J + 1e-12)
            cp = 0.5 * (cf + torch.sqrt(cf**2 + 4.0 * PHY_C_EXT**2))
            p_osm_cur = self.RT * (2.0*cp - cf - 2.0*PHY_C_EXT)
            p_osm_ref = self.RT * (torch.sqrt(fcd_ref**2 + 4.0 * PHY_C_EXT**2) - 2.0 * PHY_C_EXT)
            p_real = out[:, 3:4] + self.osmotic_ramp * (p_osm_cur - p_osm_ref)
            p_compare = out[:, 3:4] + (p_osm_cur - p_osm_ref)

            # 5. 应力张量与 PK1 映射
            s11, s22, s33 = sm11-p_real, sm22-p_real, sm33-p_real
            s12, s13, s23 = sm12, sm13, sm23
            
            if not return_pde:
                return None, None, None, None, None, (s11,s22,s33,s12,s13,s23), p_real, cf, None, wa, None, p_compare, J

            adj11, adj12, adj13 = F22*F33-F23*F32, F13*F32-F12*F33, F12*F23-F13*F22
            adj21, adj22, adj23 = F23*F31-F21*F33, F11*F33-F13*F31, F13*F21-F11*F23
            adj31, adj32, adj33 = F21*F32-F22*F31, F12*F21-F11*F32, F11*F22-F12*F21

            P = [[s11*adj11+s12*adj21+s13*adj31, s11*adj12+s12*adj22+s13*adj32, s11*adj13+s12*adj23+s13*adj33],
                 [s12*adj11+s22*adj21+s23*adj31, s12*adj12+s22*adj22+s23*adj32, s12*adj13+s22*adj23+s23*adj33],
                 [s13*adj11+s23*adj21+s33*adj31, s13*adj12+s23*adj22+s33*adj32, s13*adj13+s23*adj23+s33*adj33]]

            # 散度计算 (参考系)
            def div_row(row):
                return vgrad(row[0])[:,0:1] + vgrad(row[1])[:,1:2] + vgrad(row[2])[:,2:3]

            res_mx, res_my, res_mz = div_row(P[0]), div_row(P[1]), div_row(P[2])

            # 初始化电化学相关变量（避免LSP未绑定警告）
            k_p, pore_r, Dp = None, None, None
            gmuw, gmui = None, None
            jwx, jwy, jwz = None, None, None
            jpx, jpy, jpz = None, None, None
            
            if enable_chem:
                k_p = perm_a * torch.pow(wa/(1.0-wa+1e-8), perm_n)
                pore_r = torch.sqrt(torch.clamp(k_p, min=1e-16))
                Dp = ION_DP0 * torch.exp(-diff_a * torch.clamp(torch.pow((ION_RP/pore_r)/np.sqrt(VISCO_SI), diff_b), max=50.0))
                gmuw, gmui = vgrad(out[:,3:4]), vgrad(out[:,4:5])
                jwx, jwy, jwz = -k_p*gmuw[:,0:1], -k_p*gmuw[:,1:2], -k_p*gmuw[:,2:3]
                jpx, jpy, jpz = cp*jwx-wa*Dp*gmui[:,0:1], cp*jwy-wa*Dp*gmui[:,1:2], cp*jwz-wa*Dp*gmui[:,2:3]
                res_mw = self.chem_ramp * div_row([adj11*jwx+adj12*jwy+adj13*jwz, adj21*jwx+adj22*jwy+adj23*jwz, adj31*jwx+adj32*jwy+adj33*jwz])
                res_mi = self.chem_ramp * div_row([adj11*jpx+adj12*jpy+adj13*jpz, adj21*jpx+adj22*jpy+adj23*jpz, adj31*jpx+adj32*jpy+adj33*jpz])
            else:
                res_mw = res_mi = torch.zeros_like(res_mx)

            l_bar = torch.sum(F.relu((1.0-phi0+0.1)-J)**2) + torch.sum(F.relu(J - 1.5)**2)  # 压缩和膨胀障碍
            
            # 准备返回结果
            result = (res_mx, res_my, res_mz, res_mw, res_mi, (s11,s22,s33,s12,s13,s23), p_real, cf, None, wa, l_bar, p_compare, J)
            
            # 显式清理中间变量（帮助减少计算图引用）
            del gu, gv, gw, F11, F12, F13, F21, F22, F23, F31, F32, F33
            del B11, B22, B33, B12, B13, B23, p_el, tm
            del sm11, sm22, sm33, sm12, sm13, sm23
            if enable_chem:
                del gmuw, gmui, jwx, jwy, jwz, jpx, jpy, jpz, k_p, pore_r, Dp
            del adj11, adj12, adj13, adj21, adj22, adj23, adj31, adj32, adj33
            del P, x, out, ones
            
            self._memory_checkpoint(f"compute_physics_end (N={N})")
            return result

    def compute_residual_norm(self, x_pts: torch.Tensor, enable_chem: bool = True):
        self.model.eval()
        all_norms = []
        batch_size = 1000
        with torch.enable_grad():
            for i in range(0, x_pts.size(0), batch_size):
                xb = x_pts[i:i+batch_size].detach().clone().requires_grad_(True)
                res = self.compute_physics(xb, True, enable_chem)
                # 从计算图中分离并复制结果
                norm = torch.sqrt(res[0]**2 + res[1]**2 + res[2]**2 + res[3]**2 + res[4]**2).detach().cpu()
                all_norms.append(norm)
                # 显式清理
                del xb, res
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        self.model.train()
        # 在CPU上拼接，然后移回设备
        if all_norms:
            return torch.cat(all_norms, dim=0).to(DEVICE)
        else:
            return torch.tensor([], device=DEVICE)

    def train_step(self, n_dom: int, n_bc: int, target_disp_ratio: float = 0.0, pts=None, enable_chem: bool = True, update_weights: bool = False, epoch: int = 0):
        self._memory_checkpoint("train_step_start")
        self.optimizer.zero_grad()
        d_pts, b_pts, t_pts, s_pts, s_norm = pts["dom"], pts["btm"], pts["top"], pts["side"], pts["normals"]
        
        # 确保所有输入张量在正确设备上
        d_pts = self._device_check(d_pts, "d_pts")
        b_pts = self._device_check(b_pts, "b_pts")
        t_pts = self._device_check(t_pts, "t_pts")
        s_pts = self._device_check(s_pts, "s_pts")
        s_norm = self._device_check(s_norm, "s_norm")

        if update_weights:
            # 使用权重管理器更新权重（使用完整模型参数，更大样本量）
            sample_size = min(500, d_pts.shape[0])
            sample_points = d_pts[:sample_size]
            
            # 计算各损失项用于权重更新
            res_w = self.compute_physics(sample_points, True, enable_chem)
            out_b_w = self.model(b_pts[:min(500, b_pts.shape[0])])
            
            losses_dict = {
                "pde_mom": torch.mean(res_w[0]**2 + res_w[1]**2 + res_w[2]**2) + 1e-9,
                "pde_flow": torch.mean(res_w[3]**2) + 1e-9,
                "pde_ion": torch.mean(res_w[4]**2) + 1e-9,
                "bc": torch.mean(out_b_w[:, 0:3]**2) + 1e-9
            }
            
            # 调用权重管理器更新权重（强制更新，因为外部已决定需要更新）
            self.weight_manager.update_weights(losses_dict, self.model, epoch, sample_points, force_update=True)
            
            del res_w, out_b_w

        res = self.compute_physics(d_pts, True, enable_chem)
        w = self.weight_manager  # 指向权重管理器实例
        loss_pde = w["pde_mom"]*torch.mean(res[0]**2+res[1]**2+res[2]**2) + w["pde_flow"]*torch.mean(res[3]**2) + w["pde_ion"]*torch.mean(res[4]**2)
        loss_bar = (res[10] / d_pts.size(0)) * 1000.0 * self.barrier_ramp

        # 边界
        _, _, _, _, _, s_side, _, _, _, _, _, _, _ = self.compute_physics(s_pts, False, enable_chem)
        tx = s_side[0]*s_norm[:,0:1] + s_side[3]*s_norm[:,1:2]
        ty = s_side[3]*s_norm[:,0:1] + s_side[1]*s_norm[:,1:2]
        tz = s_side[4]*s_norm[:,0:1] + s_side[5]*s_norm[:,1:2]
        loss_side = torch.mean(tx**2+ty**2+tz**2) + self.chem_ramp*torch.mean(self.model(s_pts)[:, 3:5]**2)

        out_b, out_t = self.model(b_pts), self.model(t_pts)
        _, _, _, _, _, st, _, _, _, _, _, _, _ = self.compute_physics(t_pts, False, enable_chem)
        r_btm = torch.sqrt((b_pts[:,0]/GEO_A)**2 + (b_pts[:,1]/GEO_B)**2 + 1e-12)
        r_top = torch.sqrt((t_pts[:,0]/GEO_A)**2 + (t_pts[:,1]/GEO_B)**2 + 1e-12)
        alpha_b = (r_btm < NP_AXIS_RATIO).float().unsqueeze(1)
        alpha_t = (r_top < NP_AXIS_RATIO).float().unsqueeze(1)
        loss_chem_dir = torch.mean(alpha_b*(out_b[:,3:5]**2)) + torch.mean(alpha_t*(out_t[:,3:5]**2))

        loss_bc = w["bc"] * (torch.mean(out_b[:,0:3]**2) + torch.mean(out_t[:,0:2]**2) + torch.mean((st[2] + float(target_disp_ratio)*BC_TOP_PRESSURE)**2) + loss_side + self.chem_ramp*loss_chem_dir)

        loss_total = loss_pde + loss_bc + loss_bar
        loss_total.backward()
        self.optimizer.step()
        
        # 清理中间变量
        self._safe_cleanup(res, tx, ty, tz, out_b, out_t, s_side, st)
        if update_weights:
            # 已经在update_weights调用后清理了res_w和out_b_w
            pass
            
        self._memory_checkpoint("train_step_end")
        return loss_total.item(), loss_pde.item(), loss_bc.item()

    def debug_stats(self, x_in: torch.Tensor, epoch: int, enable_chem: bool, loss_total: float, pde_loss: float, bc_loss: float):
        with torch.no_grad():
            # 1. 获取物理全量信息 (不保留计算图)
            res = self.compute_physics(x_in[:500], True, enable_chem)
            
            # 解包: res_mx, res_my, res_mz, res_mw, res_mi, stresses, p_real, cf, _, wa, l_bar, p_compare, J_val
            rmx, rmy, rmz, rmw, rmi = res[0], res[1], res[2], res[3], res[4]
            p_comp_val = res[11].detach() * 1000.0 # MPa -> kPa
            J_val = res[12].detach()
            
            # 2. 打印中文结构化报表
            print(f"\n" + "="*70)
            print(f"[TRAIN STATS] 步数: {epoch} | 设备: {DEVICE}")
            print(f"-"*70)
            print(f"  [+] 损失统计  | 总损: {loss_total:.2e} | 物理: {pde_loss:.2e} | 边界: {bc_loss:.2e}")
            print(f"  [+] 物理残差  | 力学: {torch.sqrt(rmx**2+rmy**2+rmz**2).mean().item():.2e} | 流量: {rmw.abs().mean().item():.2e} | 离子: {rmi.abs().mean().item():.2e}")
            print(f"  [+] 权重分配  | 动量: {self.weight_manager['pde_mom'].item():.2f} | 流量: {self.weight_manager['pde_flow'].item():.2f} | 离子: {self.weight_manager['pde_ion'].item():.2f} | 边界: {self.weight_manager['bc'].item():.2f}")
            print(f"  [+] 变形对标  | J最小: {J_val.min().item():.3f} | J最大: {J_val.max().item():.3f} | J均值: {J_val.mean().item():.3f}")
            print(f"  [+] 压力对标  | P_compare(kPa) 均值: {p_comp_val.mean().item():.1f} | 最大: {p_comp_val.max().item():.1f}")
            print(f"  [+] 加载进度  | 渗透开启: {self.osmotic_ramp*100:.1f}% | 学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
            print("="*70 + "\n")

            # 3. 显存回收
            del res, J_val, p_comp_val
            torch.cuda.empty_cache()

    def save_checkpoint(self, path: str, epoch: int):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "weight_manager_state": self.weight_manager.get_state_dict(),
            "osmotic_ramp": self.osmotic_ramp,
            "chem_ramp": self.chem_ramp,
            "barrier_ramp": self.barrier_ramp
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        if not os.path.exists(path): return 0
        ckpt = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        # 加载权重管理器状态（向后兼容）
        if "weight_manager_state" in ckpt:
            self.weight_manager.load_state_dict(ckpt["weight_manager_state"])
        elif "adaptive_weights" in ckpt:
            # 旧检查点：将adaptive_weights转换为权重管理器状态
            self.weight_manager.set_weights(ckpt["adaptive_weights"])
        # 如果都没有，权重管理器将使用默认初始化
        self.osmotic_ramp = ckpt.get("osmotic_ramp", 0.0)
        self.chem_ramp = ckpt.get("chem_ramp", 0.0)
        self.barrier_ramp = ckpt.get("barrier_ramp", 0.0)
        return ckpt["epoch"]

    def train_step_lbfgs(self, n_dom, n_bc, target_disp_ratio=1.0, pts=None, enable_chem=True):
        if pts is None: return 0.0
        current_loss = [0.0]
        def closure():
            self.optimizer_lbfgs.zero_grad()
            res = self.compute_physics(pts["dom"], True, enable_chem)
            
            # L-BFGS阶段是否使用自适应权重（根据配置决定）
            if WEIGHT_LBFGS_ENABLED:
                w = self.weight_manager
                loss_pde = w["pde_mom"]*torch.mean(res[0]**2+res[1]**2+res[2]**2) + \
                          w["pde_flow"]*torch.mean(res[3]**2) + \
                          w["pde_ion"]*torch.mean(res[4]**2)
            else:
                loss_pde = torch.mean(res[0]**2+res[1]**2+res[2]**2 + res[3]**2 + res[4]**2)
            
            out_b, out_t = self.model(pts["btm"]), self.model(pts["top"])
            
            if WEIGHT_LBFGS_ENABLED:
                w = self.weight_manager
                loss_bc = w["bc"] * (torch.mean(out_b[:,0:3]**2) + torch.mean(out_t[:,0:2]**2))
            else:
                loss_bc = torch.mean(out_b[:,0:3]**2) + torch.mean(out_t[:,0:2]**2)
            
            loss = loss_pde + loss_bc
            loss.backward()
            current_loss[0] = loss.item()
            return loss
        self.optimizer_lbfgs.step(closure)
        return current_loss[0]
