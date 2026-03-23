"""
结果数据导出脚本
用于训练完成后提取关键物理场数据，供 AI 分析使用。
输出：
  - summary.txt        各场统计摘要
  - radial_z50.csv     z=H/2 径向分布（J, p, cp, Tzz, wa, FCD）
  - axial_r0.csv       r=0 轴向分布（J, p, cp, Tzz）
  - axial_r_np.csv     r=0.5a 轴向分布
  - axial_r_af.csv     r=0.9a 轴向分布
  - field_2d.csv       2D 全域采样（100x50 网格）
  - bc_check.txt       边界条件满足情况
"""
import os
import sys
import numpy as np
import torch
from torch.autograd import grad

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from solver import Solver

# ── 工具函数 ─────────────────────────────────────────────────────────────────

def compute_all_fields(solver, pts):
    """给定采样点，返回所有物理量"""
    pts = pts.to(DEVICE).requires_grad_(True)
    with torch.enable_grad():
        out = solver.model(pts)
        ones = torch.ones((pts.shape[0], 1), device=DEVICE)
        def vg(val): return grad(val, pts, ones, create_graph=True, retain_graph=True)[0]

        gu, gv, gw = vg(out[:,0:1]), vg(out[:,1:2]), vg(out[:,2:3])
        F11 = 1.0+gu[:,0:1]; F12 = gu[:,1:2]; F13 = gu[:,2:3]
        F21 = gv[:,0:1];     F22 = 1.0+gv[:,1:2]; F23 = gv[:,2:3]
        F31 = gw[:,0:1];     F32 = gw[:,1:2];     F33 = 1.0+gw[:,2:3]
        J = torch.abs(F11*(F22*F33-F23*F32) - F12*(F21*F33-F23*F31) + F13*(F21*F32-F22*F31))

        _, _, _, _, _, s_tuple, p_real, cF, _, wa, _, p_compare, J_d = \
            solver.compute_physics(pts, return_pde=False, enable_chem=True)
        s11, s22, s33, s12, s13, s23 = s_tuple

        cf   = cF
        cp   = 0.5 * (cf + torch.sqrt(cf**2 + 4.0 * PHY_C_EXT**2))
        cn   = cp - cf
        mean_s = (s11+s22+s33)/3.0
        vm = torch.sqrt(1.5*((s11-mean_s)**2+(s22-mean_s)**2+(s33-mean_s)**2
                             +2.0*(s12**2+s13**2+s23**2)))

    def n(t): return t.detach().cpu().numpy().flatten()
    return {
        "J":      n(J),
        "p_kPa":  n(p_compare) * 1000.0,
        "cp_mM":  n(cp) * 1e9,
        "cn_mM":  n(cn) * 1e9,
        "Tzz":    n(s33),
        "Trr":    n(s11),
        "VM":     n(vm),
        "wa":     n(wa),
        "cf_mM":  n(cf) * 1e9,
        "u_mm":   n(out[:,0:1]),
        "w_mm":   n(out[:,2:3]),
    }


def make_pts_radial(z_frac=0.5, n=200):
    """z=H*z_frac 截面，从 r=0 到 r=A"""
    r = torch.linspace(0, GEO_A, n)
    z = torch.full((n,), GEO_H * z_frac)
    return torch.stack([r, torch.zeros(n), z], dim=1), r.numpy()


def make_pts_axial(r_frac, n=100):
    """r=r_frac*A 轴线，从 z=0 到 z=H"""
    x = torch.full((n,), GEO_A * r_frac)
    z = torch.linspace(0, GEO_H, n)
    return torch.stack([x, torch.zeros(n), z], dim=1), z.numpy()


def make_pts_2d(nx=100, nz=50):
    """2D 网格"""
    xv = torch.linspace(0, GEO_A, nx)
    zv = torch.linspace(0, GEO_H, nz)
    X, Z = torch.meshgrid(xv, zv, indexing='ij')
    r_norm = X / GEO_A
    pts = torch.stack([X.flatten(), torch.zeros(nx*nz), Z.flatten()], dim=1)
    return pts, X.numpy(), Z.numpy(), r_norm.numpy()

# ── 主函数 ────────────────────────────────────────────────────────────────────

def export_results(model_path, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(model_path), "export")
    os.makedirs(save_dir, exist_ok=True)
    print(f"📂 输出目录: {save_dir}")

    # 加载模型
    solver = Solver()
    solver.model.load(model_path)
    solver.model.eval()
    solver.set_curriculum_params(osmotic_ramp=1.0, chem_ramp=1.0, barrier_ramp=1.0)
    print(f"✅ 模型已加载: {model_path}")

    lines = []  # summary.txt 内容

    # ── 1. 径向分布 z=H/2 ────────────────────────────────────────────────────
    pts_r, r_arr = make_pts_radial(0.5, 200)
    f_r = compute_all_fields(solver, pts_r)
    r_norm = r_arr / GEO_A

    header = "r_mm,r_norm,J,p_kPa,cp_mM,cn_mM,Tzz_MPa,Trr_MPa,VM_MPa,wa,cf_mM,u_mm"
    data_r = np.column_stack([r_arr, r_norm,
                               f_r["J"], f_r["p_kPa"], f_r["cp_mM"], f_r["cn_mM"],
                               f_r["Tzz"], f_r["Trr"], f_r["VM"],
                               f_r["wa"], f_r["cf_mM"], f_r["u_mm"]])
    np.savetxt(os.path.join(save_dir, "radial_z50.csv"), data_r,
               delimiter=',', header=header, comments='', fmt='%.6f')
    print("✅ radial_z50.csv")

    # 径向关键点统计
    idx_np  = np.argmin(np.abs(r_norm - 0.0))
    idx_if  = np.argmin(np.abs(r_norm - 0.7))
    idx_af  = np.argmin(np.abs(r_norm - 0.95))
    lines += [
        "=" * 60,
        "径向关键点 (z=H/2)",
        f"{'位置':<12} {'r_norm':>8} {'J':>8} {'p(kPa)':>10} {'cp(mM)':>8} {'Tzz(MPa)':>10} {'wa':>8}",
        f"{'NP中心':<12} {r_norm[idx_np]:>8.2f} {f_r['J'][idx_np]:>8.4f} {f_r['p_kPa'][idx_np]:>10.1f} "
        f"{f_r['cp_mM'][idx_np]:>8.1f} {f_r['Tzz'][idx_np]:>10.4f} {f_r['wa'][idx_np]:>8.4f}",
        f"{'NP/AF界面':<12} {r_norm[idx_if]:>8.2f} {f_r['J'][idx_if]:>8.4f} {f_r['p_kPa'][idx_if]:>10.1f} "
        f"{f_r['cp_mM'][idx_if]:>8.1f} {f_r['Tzz'][idx_if]:>10.4f} {f_r['wa'][idx_if]:>8.4f}",
        f"{'AF外侧':<12} {r_norm[idx_af]:>8.2f} {f_r['J'][idx_af]:>8.4f} {f_r['p_kPa'][idx_af]:>10.1f} "
        f"{f_r['cp_mM'][idx_af]:>8.1f} {f_r['Tzz'][idx_af]:>10.4f} {f_r['wa'][idx_af]:>8.4f}",
    ]

    # ── 2. 轴向分布（三条线）────────────────────────────────────────────────
    for r_frac, label, fname in [
        (0.0,  "r=0(NP中心)",   "axial_r0.csv"),
        (0.5,  "r=0.5a(NP中)",  "axial_r50.csv"),
        (0.9,  "r=0.9a(AF外)",  "axial_r90.csv"),
    ]:
        pts_a, z_arr = make_pts_axial(r_frac, 100)
        f_a = compute_all_fields(solver, pts_a)
        header_a = "z_mm,J,p_kPa,cp_mM,Tzz_MPa,VM_MPa,wa,w_mm"
        data_a = np.column_stack([z_arr, f_a["J"], f_a["p_kPa"], f_a["cp_mM"],
                                   f_a["Tzz"], f_a["VM"], f_a["wa"], f_a["w_mm"]])
        np.savetxt(os.path.join(save_dir, fname), data_a,
                   delimiter=',', header=header_a, comments='', fmt='%.6f')
        print(f"✅ {fname}")

        lines += [
            "",
            f"轴向分布 {label}",
            f"  J:      min={f_a['J'].min():.4f}  max={f_a['J'].max():.4f}  均值={f_a['J'].mean():.4f}  轴向变化={f_a['J'].max()-f_a['J'].min():.4f}",
            f"  p(kPa): min={f_a['p_kPa'].min():.1f}  max={f_a['p_kPa'].max():.1f}",
            f"  Tzz:    min={f_a['Tzz'].min():.4f}  max={f_a['Tzz'].max():.4f} MPa",
            f"  w_mm:   顶={f_a['w_mm'][-1]:.3f}  底={f_a['w_mm'][0]:.3f}",
        ]

    # ── 3. 全域统计 ──────────────────────────────────────────────────────────
    pts_2d, X2d, Z2d, R2d = make_pts_2d(100, 50)
    f_2d = compute_all_fields(solver, pts_2d)

    # NP / AF 掩码
    m_np = R2d.flatten() < NP_AXIS_RATIO
    m_af = ~m_np

    lines += [
        "",
        "=" * 60,
        "全域统计",
        f"{'场':<12} {'全局min':>10} {'全局max':>10} {'全局均值':>10} {'NP均值':>10} {'AF均值':>10}",
    ]
    for key, unit in [("J",""), ("p_kPa","kPa"), ("cp_mM","mM"), ("Tzz","MPa"), ("VM","MPa"), ("wa","")]:
        v = f_2d[key]
        lines.append(
            f"{key+unit:<12} {v.min():>10.4f} {v.max():>10.4f} {v.mean():>10.4f} "
            f"{v[m_np].mean():>10.4f} {v[m_af].mean():>10.4f}"
        )

    # 全域 CSV
    r_norm_2d = R2d.flatten()
    header_2d = "x_mm,z_mm,r_norm,J,p_kPa,cp_mM,cn_mM,Tzz_MPa,VM_MPa,wa,cf_mM,u_mm,w_mm"
    data_2d_out = np.column_stack([
        X2d.flatten(), Z2d.flatten(), r_norm_2d,
        f_2d["J"], f_2d["p_kPa"], f_2d["cp_mM"], f_2d["cn_mM"],
        f_2d["Tzz"], f_2d["VM"], f_2d["wa"], f_2d["cf_mM"],
        f_2d["u_mm"], f_2d["w_mm"]
    ])
    np.savetxt(os.path.join(save_dir, "field_2d.csv"), data_2d_out,
               delimiter=',', header=header_2d, comments='', fmt='%.6f')
    print("✅ field_2d.csv")

    # ── 4. 边界条件检查 ──────────────────────────────────────────────────────
    bc_lines = ["=" * 60, "边界条件满足情况"]

    # 底面：u=v=w=0
    n_bc = 500
    btm, top = solver.gm.sample_boundary_top_bottom(n_bc)
    side, snorm = solver.gm.sample_boundary_side(n_bc)

    with torch.no_grad():
        out_b = solver.model(btm.to(DEVICE))
        err_btm = out_b[:, 0:3].abs().mean().item()
        bc_lines.append(f"底面位移误差 (应=0):     {err_btm:.4e} mm")

        out_t = solver.model(top.to(DEVICE))
        err_top_uv = out_t[:, 0:2].abs().mean().item()
        bc_lines.append(f"顶面水平位移误差 (应=0): {err_top_uv:.4e} mm")

    # 顶面应力
    top_pts = top.to(DEVICE).requires_grad_(True)
    with torch.enable_grad():
        _, _, _, _, _, st, _, _, _, _, _, _ = solver.compute_physics(top_pts, return_pde=False)
        tzz_top = st[2].detach().cpu().numpy().flatten()
    bc_lines.append(f"顶面Tzz: 均值={tzz_top.mean():.4f} MPa  目标={-BC_TOP_PRESSURE:.4f} MPa  误差={abs(tzz_top.mean()+BC_TOP_PRESSURE):.4f}")

    # 侧边压力
    side_pts = side.to(DEVICE).requires_grad_(True)
    with torch.enable_grad():
        _, _, _, _, _, _, _, _, _, _, _, p_side = solver.compute_physics(side_pts, return_pde=False)
        p_side_kpa = p_side.detach().cpu().numpy().flatten() * 1000.0
    bc_lines.append(f"侧边压力 (应≈0): 均值={p_side_kpa.mean():.1f} kPa  最大={p_side_kpa.max():.1f} kPa")

    lines += bc_lines

    # ── 5. 写入 summary.txt ──────────────────────────────────────────────────
    summary_path = os.path.join(save_dir, "summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✅ summary.txt")
    print('\n'.join(lines))

    print(f"\n🎯 全部导出完成 → {save_dir}")
    return save_dir


if __name__ == "__main__":
    import glob

    # 自动查找最新的模型文件
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        candidates = sorted(glob.glob(os.path.join(OUTPUT_BASE_DIR, "**", "model_final.pth"), recursive=True))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(OUTPUT_BASE_DIR, "**", "*.pth"), recursive=True))
        if not candidates:
            print("❌ 未找到模型文件，请指定路径: python export_results.py <model.pth>")
            sys.exit(1)
        model_path = candidates[-1]
        print(f"🔍 自动选择最新模型: {model_path}")

    export_results(model_path)
