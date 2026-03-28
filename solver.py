"""
物理求解器模块 - 终极稳健 Windows 优化版 v7 (J单调性增强版)
1. 严格落实 J 分布：内部 (r<0.7) 均匀平均，外部 (r>0.7) 径向单调递增。
2. 修复 J 限制逻辑，确保 wa 物理安全 (下限 0.3)。
3. 增强数值稳定性，解决拐角奇异点。
"""
from __future__ import annotations
import os
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
        self.osmotic_ramp, self.chem_ramp, self.barrier_ramp = 0.0, 0.0, 1.0
        self.weights = {name: torch.tensor(val, device=DEVICE) for name, val in WEIGHT_INIT.items()}
        print(f"求解器初始化完成: Windows稳健优化版 v7 (J-Monotonic)")

    def _update_weights(self, losses_dict: dict, epoch: int):
        if WEIGHT_STRATEGY == "fixed" or epoch % WEIGHT_UPDATE_FREQ != 0: return
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        grads_norm = {}
        for name, loss in losses_dict.items():
            if not (isinstance(loss, torch.Tensor) and loss.requires_grad): continue
            try:
                g_list = grad(loss, model_params, retain_graph=True, allow_unused=True)
                gn = torch.sqrt(torch.sum(torch.stack([torch.norm(g.detach())**2 for g in g_list if g is not None])) + 1e-8)
                grads_norm[name] = gn
            except: continue
        if "pde_mom" in grads_norm:
            base_gn = grads_norm["pde_mom"]
            for name in self.weights.keys():
                if name in grads_norm and name != "pde_mom":
                    target_w = torch.clamp(base_gn / (grads_norm[name] + 1e-8), *WEIGHT_CLAMP_RANGES.get(name, (0.01, 100.0)))
                    self.weights[name] = WEIGHT_MOMENTUM * self.weights[name] + (1 - WEIGHT_MOMENTUM) * target_w
        torch.cuda.empty_cache()

    def set_curriculum_params(self, osmotic_ramp=None, chem_ramp=None, barrier_ramp=None):
        if osmotic_ramp is not None: self.osmotic_ramp = float(osmotic_ramp)
        if chem_ramp is not None: self.chem_ramp = float(chem_ramp)
        if barrier_ramp is not None: self.barrier_ramp = float(barrier_ramp)

    def compute_physics(self, x_in: torch.Tensor, return_pde: bool = True, enable_chem: bool = True):
        with torch.enable_grad():
            N = x_in.shape[0]
            x = x_in.detach().clone().requires_grad_(True)
            out = self.model(x)
            ones = torch.ones((N, 1), device=DEVICE)

            # 基础梯度
            gu = grad(out[:, 0:1], x, ones, create_graph=True)[0]
            gv = grad(out[:, 1:2], x, ones, create_graph=True)[0]
            gw = grad(out[:, 2:3], x, ones, create_graph=True)[0]
            F11, F12, F13 = 1.0+gu[:,0:1], gu[:,1:2], gu[:,2:3]
            F21, F22, F23 = gv[:,0:1], 1.0+gv[:,1:2], gv[:,2:3]
            F31, F32, F33 = gw[:,0:1], gw[:,1:2], 1.0+gw[:,2:3]
            J = torch.abs(F11*(F22*F33-F23*F32) - F12*(F21*F33-F23*F31) + F13*(F21*F32-F22*F31)) + 1e-6
            invJ = torch.clamp(1.0 / J, max=10.0)

            mu_s, lam_s, phi0, fcd_ref, perm_a, perm_n, diff_a, diff_b, _ = self.gm.get_material_params(x[:,0:1], x[:,1:2])
            wa = torch.clamp(1.0 - (1.0 - phi0)*invJ, 0.3, 0.99)
            
            # --- J 空间分布正则化 (对标 COMSOL 趋势) ---
            gJ = grad(J, x, ones, create_graph=True)[0]
            dJdx, dJdy, dJdz = gJ[:,0:1], gJ[:,1:2], gJ[:,2:3]
            r_vec = torch.sqrt(x[:,0:1]**2 + x[:,1:2]**2 + 1e-12)
            dJdr = (x[:,0:1]*dJdx + x[:,1:2]*dJdy) / (r_vec + 1e-8)
            
            r_norm = torch.sqrt((x[:,0:1]/GEO_A)**2 + (x[:,1:2]/GEO_B)**2 + 1e-12)
            # 髓核区域 (r < 0.7): 强制均匀 (梯度为0)
            mask_np = torch.sigmoid(30.0 * (0.7 - r_norm))
            l_j_uniform = mask_np * (dJdx**2 + dJdy**2 + dJdz**2)
            
            # 纤维环区域 (r > 0.7): 强制径向递增 (dJ/dr > 0)
            mask_af = torch.sigmoid(30.0 * (r_norm - 0.7))
            l_j_monotonic = mask_af * F.relu(-dJdr + 0.05) # 惩罚下降或平坦，强制上升
            
            # 安全阀障碍
            J_limit = (1.0 - phi0) / 0.7
            l_bar_safe = F.relu(J_limit + 0.02 - J)**2
            
            # 总物理正则项
            l_phys_reg = l_bar_safe + 1.0 * (l_j_uniform + l_j_monotonic)

            # 3. 弹性与纤维
            B11, B22, B33 = F11**2+F12**2+F13**2, F21**2+F22**2+F23**2, F31**2+F32**2+F33**2
            B12, B13, B23 = F11*F21+F12*F22+F13*F23, F11*F31+F12*F32+F13*F33, F21*F31+F22*F32+F23*F33
            p_el = lam_s * phi0 * (J - 1.0) / torch.clamp(J - (1.0 - phi0), min=0.005)
            tm = mu_s * invJ
            sm11, sm22, sm33 = tm*(B11-1)+p_el, tm*(B22-1)+p_el, tm*(B33-1)+p_el
            sm12, sm13, sm23 = tm*B12, tm*B13, tm*B23
            if ENABLE_ANISOTROPY:
                beta = np.deg2rad(MAT_FIBER_THETA_INNER)
                r_c = torch.sqrt(x[:,0:1]**2 + x[:,1:2]**2 + 1e-9); etx, ety = -x[:,1:2]/r_c, x[:,0:1]/r_c
                cos_b, sin_b = np.cos(beta), np.sin(beta)
                t_af = torch.clamp((torch.sqrt((x[:,0:1]/GEO_A)**2 + (x[:,1:2]/GEO_B)**2 + 1e-12) - self.gm.np_ratio)/(1-self.gm.np_ratio+1e-8), 0, 1)
                m_af_w = (3*t_af**2 - 2*t_af**3) * (t_af > 0).float()
                for sgn in [1.0, -1.0]:
                    a0x, a0y, a0z = sgn*sin_b*etx, sgn*sin_b*ety, cos_b
                    nfx, nfy, nfz = F11*a0x+F12*a0y+F13*a0z, F21*a0x+F22*a0y+F23*a0z, F31*a0x+F32*a0y+F33*a0z
                    EN = 0.5 * (nfx**2 + nfy**2 + nfz**2 - 1.0)
                    fs = MAT_FIBER_ALPHA_MAX * t_af * EN * (EN > 0).float() * invJ * m_af_w
                    sm11, sm22, sm33 = sm11+fs*nfx**2, sm22+fs*nfy**2, sm33+fs*nfz**2
                    sm12, sm13, sm23 = sm12+fs*nfx*nfy, sm13+fs*nfx*nfz, sm23+fs*nfy*nfz
            cf = fcd_ref * phi0 / (wa * J + 1e-12); cp = 0.5 * (cf + torch.sqrt(cf**2 + 4.0 * PHY_C_EXT**2))
            p_osm_cur = self.RT * (2.0*cp - cf - 2.0*PHY_C_EXT); p_osm_ref = self.RT * (torch.sqrt(fcd_ref**2 + 4.0 * PHY_C_EXT**2) - 2.0 * PHY_C_EXT)
            p_real = out[:, 3:4] + self.osmotic_ramp * (p_osm_cur - p_osm_ref); p_compare = out[:, 3:4] + (p_osm_cur - p_osm_ref)
            s11, s22, s33, s12, s13, s23 = sm11-p_real, sm22-p_real, sm33-p_real, sm12, sm13, sm23
            c11, c12, c13 = F22*F33-F23*F32, F23*F31-F21*F33, F21*F32-F22*F31
            c21, c22, c23 = F13*F32-F12*F33, F11*F33-F13*F31, F12*F21-F11*F32
            c31, c32, c33 = F12*F23-F13*F22, F13*F21-F11*F23, F11*F22-F12*F21
            if not return_pde:
                zeros = torch.zeros((N, 1), device=DEVICE)
                return (zeros, zeros, zeros, zeros, zeros, (s11,s22,s33,s12,s13,s23), p_real, cf, wa, l_phys_reg, p_compare, J)
            p11, p12, p13 = s11*c11 + s12*c21 + s13*c31, s11*c12 + s12*c22 + s13*c32, s11*c13 + s12*c23 + s13*c33
            p21, p22, p23 = s12*c11 + s22*c21 + s23*c31, s12*c12 + s22*c22 + s23*c32, s12*c13 + s22*c23 + s23*c33
            p31, p32, p33 = s13*c11 + s23*c21 + s33*c31, s13*c12 + s23*c22 + s33*c32, s13*c13 + s23*c23 + s33*c33
            res_mx = grad(p11, x, ones, create_graph=True)[0][:,0:1] + grad(p12, x, ones, create_graph=True)[0][:,1:2] + grad(p13, x, ones, create_graph=True)[0][:,2:3]
            res_my = grad(p21, x, ones, create_graph=True)[0][:,0:1] + grad(p22, x, ones, create_graph=True)[0][:,1:2] + grad(p23, x, ones, create_graph=True)[0][:,2:3]
            res_mz = grad(p31, x, ones, create_graph=True)[0][:,0:1] + grad(p32, x, ones, create_graph=True)[0][:,1:2] + grad(p33, x, ones, create_graph=True)[0][:,2:3]
            if enable_chem:
                k_p = perm_a * torch.pow(wa/(1.0-wa+1e-8), perm_n)
                gmuw, gmui = grad(out[:,3:4], x, ones, create_graph=True)[0], grad(out[:,4:5], x, ones, create_graph=True)[0]
                jwx, jwy, jwz = -k_p*gmuw[:,0:1], -k_p*gmuw[:,1:2], -k_p*gmuw[:,2:3]
                jpx, jpy, jpz = cp*jwx-wa*Dp*gmui[:,0:1], cp*jwy-wa*Dp*gmui[:,1:2], cp*jwz-wa*Dp*gmui[:,2:3]
                JwX, JwY, JwZ = c11*jwx+c21*jwy+c31*jwz, c12*jwx+c22*jwy+c32*jwz, c13*jwx+c23*jwy+c33*jwz
                JpX, JpY, JpZ = c11*jpx+c21*jpy+c31*jpz, c12*jpx+c22*jpy+c32*jpz, c13*jpx+c23*jpy+c33*jpz
                res_mw = self.chem_ramp * (grad(JwX, x, ones, create_graph=True)[0][:,0:1] + grad(JwY, x, ones, create_graph=True)[0][:,1:2] + grad(JwZ, x, ones, create_graph=True)[0][:,2:3])
                res_mi = self.chem_ramp * (grad(JpX, x, ones, create_graph=True)[0][:,0:1] + grad(JpY, x, ones, create_graph=True)[0][:,1:2] + grad(JpZ, x, ones, create_graph=True)[0][:,2:3])
            else: res_mw = res_mi = torch.zeros_like(res_mx)
            return (res_mx, res_my, res_mz, res_mw, res_mi, (s11,s22,s33,s12,s13,s23), p_real, cf, wa, l_phys_reg, p_compare, J)

    def compute_residual_norm(self, x_pts: torch.Tensor, enable_chem=True):
        self.model.eval(); all_norms = []
        with torch.enable_grad():
            for i in range(0, x_pts.size(0), 1000):
                res = self.compute_physics(x_pts[i:i+1000], True, enable_chem)
                norm = torch.sqrt(res[0].detach()**2 + res[1].detach()**2 + res[2].detach()**2).cpu()
                all_norms.append(norm)
        self.model.train(); return torch.cat(all_norms, 0).to(DEVICE)

    def train_step(self, n_dom, n_bc, target_disp_ratio=0.0, pts=None, enable_chem=True, update_weights=False, epoch=0):
        self.model.train(); self.optimizer.zero_grad()
        d_pts = pts["dom"].to(DEVICE); b_pts = pts["btm"].to(DEVICE); t_pts = pts["top"].to(DEVICE); s_pts = pts["side"].to(DEVICE); s_norm = pts["normals"].to(DEVICE)
        if update_weights:
            res_w = self.compute_physics(d_pts[torch.randperm(d_pts.size(0))[:500]], True, enable_chem)
            out_b_w = self.model(b_pts[:500])
            self._update_weights({"pde_mom": torch.mean(res_w[0]**2+res_w[1]**2+res_w[2]**2)+1e-9, "pde_flow": torch.mean(res_w[3]**2)+1e-9, "bc": torch.mean(out_b_w[:,0:3]**2)+1e-9}, epoch)
        res = self.compute_physics(d_pts, True, enable_chem)
        loss_pde = self.weights["pde_mom"]*torch.mean(res[0]**2+res[1]**2+res[2]**2) + self.weights["pde_flow"]*torch.mean(res[3]**2) + self.weights["pde_ion"]*torch.mean(res[4]**2)
        loss_bar = torch.mean(res[9]) * 100.0 * self.barrier_ramp
        res_side = self.compute_physics(s_pts, False, enable_chem); s_side = res_side[5]
        tx, ty, tz = s_side[0]*s_norm[:,0:1]+s_side[3]*s_norm[:,1:2], s_side[3]*s_norm[:,0:1]+s_side[1]*s_norm[:,1:2], s_side[4]*s_norm[:,0:1]+s_side[5]*s_norm[:,1:2]
        loss_side = torch.mean(tx**2+ty**2+tz**2) + self.chem_ramp*torch.mean(self.model(s_pts)[:, 3:5]**2)
        out_b, out_t = self.model(b_pts), self.model(t_pts)
        res_t = self.compute_physics(t_pts, False, enable_chem); szz_t = res_t[5][2]
        loss_bc = self.weights["bc"] * (torch.mean(out_b[:,0:3]**2) + torch.mean(out_t[:,0:2]**2) + torch.mean((szz_t+target_disp_ratio*BC_TOP_PRESSURE)**2) + loss_side + self.chem_ramp*(torch.mean(out_b[:,3:5]**2)+torch.mean(out_t[:,3:5]**2)))
        loss = loss_pde + loss_bc + loss_bar
        loss.backward(); self.optimizer.step(); torch.cuda.empty_cache()
        return loss.item(), loss_pde.item(), loss_bc.item()

    def debug_stats(self, x_in, epoch, enable_chem, loss_total, pde_loss, bc_loss):
        with torch.no_grad():
            res = self.compute_physics(x_in[:500], True, enable_chem)
            szz_avg, wa_avg = res[5][2].mean().item(), res[8].mean().item()
            p_comp_val, J_val = res[10].detach()*1000.0, res[11].detach()
            print(f"\n" + "="*80 + f"\n[EPOCH {epoch:5d}] 总损: {loss_total:.2e} | 物理: {pde_loss:.2e} | 边界: {bc_loss:.2e}\n" + "-"*80)
            print(f" PDE残差 | 力学: {torch.sqrt(res[0]**2+res[1]**2+res[2]**2).mean():.2e} | 流量: {res[3].abs().mean():.2e} | 离子: {res[4].abs().mean():.2e}")
            print(f" 物理状态 | J均值: {J_val.mean():.3f} [Min:{J_val.min():.3f}] | wa: {wa_avg:.3f} | Szz: {szz_avg:.3f} MPa")
            print(f" 压力对标 | P均值: {p_comp_val.mean():.1f} kPa | P最大: {p_comp_val.max():.1f} kPa")
            print(f" 权重分配 | Mom: {self.weights['pde_mom']:.2f} | Flow: {self.weights['pde_flow']:.2f} | BC: {self.weights['bc']:.2f}")
            print(f" 加载进度 | 渗透压加载: {self.osmotic_ramp*100:.1f}% | 学习率: {self.optimizer.param_groups[0]['lr']:.2e}\n" + "="*80 + "\n")

    def save_checkpoint(self, path, epoch):
        torch.save({"epoch": epoch, "model_state": self.model.state_dict(), "optimizer_state": self.optimizer.state_dict(), "weights": {n: w.cpu() for n, w in self.weights.items()}, "osmotic_ramp": self.osmotic_ramp, "chem_ramp": self.chem_ramp}, path)

    def load_checkpoint(self, path):
        if not os.path.exists(path): return 0
        ckpt = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        if "weights" in ckpt: self.weights = {n: w.to(DEVICE) for n, w in ckpt["weights"].items()}
        self.osmotic_ramp, self.chem_ramp = ckpt.get("osmotic_ramp", 0.0), ckpt.get("chem_ramp", 0.0)
        return ckpt["epoch"]

    def train_step_lbfgs(self, n_dom, n_bc, target_disp_ratio=1.0, pts=None, enable_chem=True):
        loss_storage = [0.0]
        def closure():
            self.optimizer_lbfgs.zero_grad()
            d_pts = pts["dom"].to(DEVICE); b_pts = pts["btm"].to(DEVICE); t_pts = pts["top"].to(DEVICE); s_pts = pts["side"].to(DEVICE); s_norm = pts["normals"].to(DEVICE)
            res = self.compute_physics(d_pts, True, enable_chem)
            l_pde = self.weights["pde_mom"]*torch.mean(res[0]**2+res[1]**2+res[2]**2) + self.weights["pde_flow"]*torch.mean(res[3]**2) + self.weights["pde_ion"]*torch.mean(res[4]**2)
            l_bar = torch.mean(res[9]) * 100.0
            res_side = self.compute_physics(s_pts, False, enable_chem); s_side = res_side[5]
            tx, ty, tz = s_side[0]*s_norm[:,0:1]+s_side[3]*s_norm[:,1:2], s_side[3]*s_norm[:,0:1]+s_side[1]*s_norm[:,1:2], s_side[4]*s_norm[:,0:1]+s_side[5]*s_norm[:,1:2]
            l_side = torch.mean(tx**2+ty**2+tz**2) + self.chem_ramp*torch.mean(self.model(s_pts)[:, 3:5]**2)
            out_b, out_t = self.model(b_pts), self.model(t_pts)
            res_t = self.compute_physics(t_pts, False, enable_chem); szz_t = res_t[5][2]
            l_bc = self.weights["bc"] * (torch.mean(out_b[:,0:3]**2) + torch.mean(out_t[:,0:2]**2) + torch.mean((szz_t+target_disp_ratio*BC_TOP_PRESSURE)**2) + l_side + self.chem_ramp*(torch.mean(out_b[:,3:5]**2)+torch.mean(out_t[:,3:5]**2)))
            total_l = l_pde + l_bc + l_bar
            total_l.backward(); loss_storage[0] = total_l.item(); return total_l
        self.optimizer_lbfgs.step(closure); return loss_storage[0]
