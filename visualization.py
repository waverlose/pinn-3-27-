"""
可视化模块 - 终极合并版 (Visualization Module - Ultimate Merged v4)
包含: 训练监控、中文对标图表、全量 VTK 导出、变形网格。
修复: 3D VTK Different item sizes 错误, 向量场导出逻辑, 兼容 suffix 参数。
"""
import os
# 关键：在导入 plt 之前设置环境变量并指定非 GUI 后端
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
# 使用 Agg 后端，防止 GUI 线程错误
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch.autograd import grad
from matplotlib.gridspec import GridSpec
from config import *

# 配置中文字体与符号
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False 
plt.rcParams['axes.unicode_minus'] = False

def export_revolved_vtk(save_dir, data_2d, n_theta=36):
    """
    将 2D 轴对称解旋转为 3D 圆柱体 VTK
    """
    print(f"🔄 正在生成 3D 旋转全模型 VTK (N_theta={n_theta})...")
    try:
        from pyevtk.hl import gridToVTK
        
        R_2d = data_2d["X"] 
        Z_2d = data_2d["Z"] 
        nx, nz = R_2d.shape
        theta = np.linspace(0, 2*np.pi, n_theta)
        
        # 1. 构建 3D 坐标网格 (nx, n_theta, nz)
        # 使用 meshgrid 确保形状完全一致且连续
        # 先准备 1D 向量
        rs = R_2d[:, 0] # 径向分布
        zs = Z_2d[0, :] # 轴向分布
        
        # R_3d, T_3d, Z_3d 形状均为 (nx, n_theta, nz)
        R_3d, T_3d, Z_3d_coords = np.meshgrid(rs, theta, zs, indexing='ij')
        
        X_3d = np.ascontiguousarray(R_3d * np.cos(T_3d), dtype=np.float32)
        Y_3d = np.ascontiguousarray(R_3d * np.sin(T_3d), dtype=np.float32)
        Z_3d = np.ascontiguousarray(Z_3d_coords, dtype=np.float32)
        
        # 2. 准备点数据
        pointData = {}
        
        # 处理标量: (nx, nz) -> (nx, n_theta, nz)
        scalars = ["P_comsol", "P_compare", "Phi", "J", "FCD", "Cation", "Alpha", "S_VM"]
        for key in scalars:
            if key in data_2d:
                # 广播 2D 数据到 3D
                val_2d = data_2d[key]
                val_3d = np.zeros((nx, n_theta, nz), dtype=np.float32)
                for t in range(n_theta):
                    val_3d[:, t, :] = val_2d
                pointData[key] = np.ascontiguousarray(val_3d, dtype=np.float32)
        
        # 处理位移向量 (转换为笛卡尔分量)
        ur_2d = data_2d["U_r"]
        wz_2d = data_2d["W_z"]
        
        ux_3d = np.zeros((nx, n_theta, nz), dtype=np.float32)
        uy_3d = np.zeros((nx, n_theta, nz), dtype=np.float32)
        uz_3d = np.zeros((nx, n_theta, nz), dtype=np.float32)
        
        for t in range(n_theta):
            angle = theta[t]
            ux_3d[:, t, :] = ur_2d * np.cos(angle)
            uy_3d[:, t, :] = ur_2d * np.sin(angle)
            uz_3d[:, t, :] = wz_2d
            
        # 导出为向量场 (ParaView 识别)
        pointData["Displacement"] = (np.ascontiguousarray(ux_3d, dtype=np.float32), 
                                     np.ascontiguousarray(uy_3d, dtype=np.float32), 
                                     np.ascontiguousarray(uz_3d, dtype=np.float32))
        
        # 导出为独立标量 (保险方案)
        pointData["U_x"] = np.ascontiguousarray(ux_3d, dtype=np.float32)
        pointData["U_y"] = np.ascontiguousarray(uy_3d, dtype=np.float32)
        pointData["U_z"] = np.ascontiguousarray(uz_3d, dtype=np.float32)

        # 3. 写入文件
        vtk_path = os.path.join(save_dir, "ivd_3d_full")
        gridToVTK(vtk_path, X_3d, Y_3d, Z_3d, pointData=pointData)
        print(f"✅ 3D 全模型导出成功: {vtk_path}.vts")
        
    except Exception as e:
        print(f"⚠️ 3D VTK 导出失败: {e}")
        # 打印调试信息
        try: print(f"   Debug Shapes: X_3d={X_3d.shape}, P_comsol={pointData.get('P_comsol', 'N/A').shape}")
        except: pass

def generate_comsol_benchmark_figures(solver, save_dir, suffix="", gen_deformed=True, gen_undeformed=True, gen_vtk=True):
    """
    生成对标图表并导出 VTK (输出到 save_dir)
    """
    print(f"\n[Post-Process]  {suffix}...")
    
    # 1. 准备网格
    # 确保 ramp 参数已达到稳态以进行可视化
    old_params = (solver.osmotic_ramp, solver.chem_ramp, solver.barrier_ramp)
    solver.set_curriculum_params(osmotic_ramp=1.0, chem_ramp=1.0, barrier_ramp=1.0)
    
    nx, nz = 100, 100
    x_v = torch.linspace(0, GEO_A, nx)
    z_v = torch.linspace(0, GEO_H, nz)
    X, Z = torch.meshgrid(x_v, z_v, indexing='ij')
    pts = torch.stack([X.flatten(), torch.zeros_like(X.flatten()), Z.flatten()], dim=1).to(DEVICE)
    pts.requires_grad_(True)
    
    # 2. 物理场推断
    with torch.enable_grad():
        out = solver.model(pts)
        u, v, w = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        
        # --- 核心修正：直接从位移梯度计算 J (Jacobian) ---
        ones_v = torch.ones((pts.shape[0], 1), device=DEVICE)
        def vgrad_v(val): return grad(val, pts, ones_v, create_graph=True, retain_graph=True)[0]
        gu, gv, gw = vgrad_v(u), vgrad_v(v), vgrad_v(w)
        F11, F12, F13 = 1.0+gu[:,0:1], gu[:,1:2], gu[:,2:3]
        F21, F22, F23 = gv[:,0:1], 1.0+gv[:,1:2], gv[:,2:3]
        F31, F32, F33 = gw[:,0:1], gw[:,1:2], 1.0+gw[:,2:3]
        J_direct = torch.abs(F11*(F22*F33-F23*F32) - F12*(F21*F33-F23*F31) + F13*(F21*F32-F22*F31))

        # 推断其他物理场 - 修正解包至 13 位
        _, _, _, _, _, s_tuple, p_real, cF, k_perm, wa_phys, _, p_compare_calc, _ = solver.compute_physics(pts, return_pde=False, enable_chem=True)
        # 修正解包顺序: (s11, s22, s33, s12, s13, s23)
        s11, s22, s33, s12, s13, s23 = s_tuple
        # 补全 S_theta (Hoop Stress): 轴对称下 s22 对应 s_theta_theta
        s_theta = s22
        
        _, _, phi0, fcd0, _, _, _, _, alpha_map = solver.gm.get_material_params(pts[:, 0:1], pts[:, 1:2])
        c_pos = 0.5 * (cF + torch.sqrt(cF**2 + 4 * PHY_C_EXT**2))        
        # Von Mises
        mean_s = (s11 + s22 + s33) / 3.0
        vm = torch.sqrt(1.5 * ((s11-mean_s)**2 + (s22-mean_s)**2 + (s33-mean_s)**2 + 2.0*(s12**2 + s13**2 + s23**2)))

    # 3. 数据转 Numpy
    def tnp(t): return t.detach().cpu().numpy().reshape(nx, nz)
    data_dict = {
        "X": tnp(X), "Z": tnp(Z), "U_r": tnp(u), "W_z": tnp(w),
        "S_rr": tnp(s11), "S_zz": tnp(s33), "S_rz": tnp(s13), "S_theta": tnp(s_theta), "S_VM": tnp(vm),
        "P_pore": tnp(p_compare_calc) * 1000.0, # 统一为全量孔隙压力 (kPa)
        "Phi": tnp(wa_phys), "J_solid": tnp(J_direct), "J": tnp(J_direct),
        "FCD": tnp(cF) * 1e9, "Cation": tnp(c_pos) * 1e9, "Alpha": tnp(alpha_map)
    }

    if gen_vtk:
        # 4. 导出 2D VTK
        try:
            from pyevtk.hl import gridToVTK
            xv = np.zeros((nx, 1, nz)); xv[:, 0, :] = data_dict["X"]
            yv = np.zeros((nx, 1, nz))
            zv = np.zeros((nx, 1, nz)); zv[:, 0, :] = data_dict["Z"]
            pData = {k: v.reshape(nx, 1, nz) for k, v in data_dict.items() if k not in ["X", "Z"]}
            pData["Displacement"] = (np.ascontiguousarray(pData["U_r"]), 
                                     np.ascontiguousarray(np.zeros_like(pData["U_r"])), 
                                     np.ascontiguousarray(pData["W_z"]))
            gridToVTK(os.path.join(save_dir, f"ivd_slice_xz{suffix}"), xv, yv, zv, pointData=pData)
        except Exception as e:
            print(f"⚠️ 2D VTK 导出跳过: {e}")

        # 5. 导出 3D VTK
        export_revolved_vtk(save_dir, data_dict)

    # 6. 绘制图表
    def_dir = os.path.join(save_dir, "Deformed_Plots")
    undef_dir = os.path.join(save_dir, "Undeformed_Plots")
    csv_dir = os.path.join(save_dir, "CSV_Data")
    if gen_deformed: os.makedirs(def_dir, exist_ok=True)
    if gen_undeformed: os.makedirs(undef_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # 6. 核心绘图逻辑：COMSOL 风格渲染 (双重输出)
    def comsol_style_plot(name, data, title, unit, cmap='jet', levels=100):
        vmin, vmax = np.nanpercentile(data, 0.5), np.nanpercentile(data, 99.5)
        if vmin == vmax: vmin -= 0.1; vmax += 0.1
        
        loop_list = []
        if gen_undeformed: loop_list.append((False, undef_dir))
        if gen_deformed: loop_list.append((True, def_dir))
        
        for is_deformed, target_dir in loop_list:
            plt.figure(figsize=(9, 7), dpi=300)
            ax = plt.gca()
            
            if is_deformed:
                X_plot = data_dict["X"] + data_dict["U_r"]
                Z_plot = data_dict["Z"] + data_dict["W_z"]
            else:
                X_plot = data_dict["X"]
                Z_plot = data_dict["Z"]
            
            im = plt.contourf(X_plot, Z_plot, data, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
            
            skip = 5 
            plt.plot(X_plot[::skip, :].T, Z_plot[::skip, :].T, color='black', lw=0.15, alpha=0.3)
            plt.plot(X_plot[:, ::skip], Z_plot[:, ::skip], color='black', lw=0.15, alpha=0.3)
            
            plt.contour(X_plot, Z_plot, data_dict["Alpha"], levels=[0.5], colors='white', linestyles='-', linewidths=1.5, alpha=0.8)
            
            plt.plot(X_plot[-1, :], Z_plot[-1, :], 'k-', lw=2.0)
            plt.plot(X_plot[:, 0], Z_plot[:, 0], 'k-', lw=2.0)
            plt.plot(X_plot[:, -1], Z_plot[:, -1], 'k-', lw=2.0)
            
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label(f"{title} ({unit})", fontsize=12, labelpad=10)
            
            plt.title(f"椎间盘 {title} - {suffix}", fontsize=14, fontweight='bold', pad=15)
            plt.xlabel("径向 R (mm)", fontsize=12); plt.ylabel("轴向 Z (mm)", fontsize=12)
            plt.axis('equal')
            
            for spine in ax.spines.values(): spine.set_visible(False)
            plt.grid(True, linestyle=':', alpha=0.2)
            
            plt.tight_layout()
            plt.savefig(os.path.join(target_dir, f"{name}{suffix}.png"), bbox_inches='tight')
            plt.close()

    # 执行全量对标图表生成
    print("🎨 Generating COMSOL-style contour plots...")
    comsol_style_plot("1_位移_径向_Ur", data_dict["U_r"], "径向位移 (Ur)", "mm", 'rainbow')
    comsol_style_plot("2_位移_轴向_Wz", data_dict["W_z"], "轴向位移 (Wz)", "mm", 'rainbow')
    comsol_style_plot("3_应力_轴向_Szz", data_dict["S_zz"], "轴向 Cauchy 应力 (Szz)", "MPa", 'viridis')
    comsol_style_plot("3_应力_径向_Srr", data_dict["S_rr"], "径向 Cauchy 应力 (Srr)", "MPa", 'viridis')
    comsol_style_plot("3_应力_环向_Stheta", data_dict["S_theta"], "环向 Cauchy 应力 (Sθ)", "MPa", 'viridis')
    comsol_style_plot("3_应力_剪切_Srz", data_dict["S_rz"], "剪切应力 (Srz)", "MPa", 'RdYlBu_r')
    comsol_style_plot("4_应力_VM", data_dict["S_VM"], "Von-Mises 等效应力", "MPa", 'magma')
    comsol_style_plot("5_孔隙压力_Ppore", data_dict["P_pore"], "总孔隙压力 (Ppore)", "kPa", 'turbo')
    comsol_style_plot("6_含水量_wa", data_dict["Phi"], "实时含水量 (Porosity)", "-", 'Blues')
    comsol_style_plot("6_体积比_J_solid", data_dict["J_solid"], "固相体积比 (J_solid)", "-", 'Spectral_r')
    comsol_style_plot("7_固定电荷密度_FCD", data_dict["FCD"], "固定电荷密度 (FCD)", "mM", 'YlGn')
    comsol_style_plot("8_阳离子浓度_cp", data_dict["Cation"], "阳离子浓度 (cp)", "mM", 'RdPu')
    
    # 7. 变形网格图
    plt.figure(figsize=(8, 7), dpi=200)
    X_def, Z_def = data_dict["X"] + data_dict["U_r"], data_dict["Z"] + data_dict["W_z"]
    skip = 8
    for j in range(0, nz, skip): plt.plot(data_dict["X"][:, j], data_dict["Z"][:, j], 'k--', lw=0.5, alpha=0.2)
    for i in range(0, nx, skip): plt.plot(data_dict["X"][i, :], data_dict["Z"][i, :], 'k--', lw=0.5, alpha=0.2)
    for j in range(0, nz, skip): plt.plot(X_def[:, j], Z_def[:, j], 'r-', lw=0.8, alpha=0.5)
    for i in range(0, nx, skip): plt.plot(X_def[i, :], Z_def[i, :], 'b-', lw=0.8, alpha=0.5)
    plt.title(f"椎间盘大变形网格图 {suffix}"); plt.axis('equal'); plt.savefig(os.path.join(def_dir, f"9_变形网格{suffix}.png")); plt.close()

    # 还原 solver 状态
    solver.set_curriculum_params(osmotic_ramp=old_params[0], chem_ramp=old_params[1], barrier_ramp=old_params[2])

def plot_loss_history(loss_history, save_dir):
    plt.figure(figsize=(10, 6))
    for k, v in loss_history.items():
        if v: plt.semilogy(v, label=k, alpha=0.8)
    plt.title("训练损失历史 (Loss History)"); plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'loss_history.png'), dpi=200); plt.close()

def plot_j_radial(solver, save_dir, suffix=""):
    """
    绘制 J 沿径向的分布曲线 (z=H/2) 并导出 CSV
    """
    print(f"📈 正在生成 J 径向分布图与 CSV 导出...")
    
    # 1. 连续曲线采样 (200 点)
    r_norm_curve = np.linspace(0, 1.0, 200)
    x_curve = r_norm_curve * GEO_A
    z_fixed = GEO_H / 2.0
    pts_curve = torch.stack([
        torch.tensor(x_curve, dtype=torch.float32, device=DEVICE),
        torch.zeros(200, device=DEVICE),
        torch.full((200,), z_fixed, device=DEVICE)
    ], dim=1).requires_grad_(True)
    
    # 2. 离散采样点 (21 点: 0, 0.05, ..., 1.0)
    r_norm_samp = np.arange(0, 1.05, 0.05)
    x_samp = r_norm_samp * GEO_A
    pts_samp = torch.stack([
        torch.tensor(x_samp, dtype=torch.float32, device=DEVICE),
        torch.zeros(len(x_samp), device=DEVICE),
        torch.full((len(x_samp),), z_fixed, device=DEVICE)
    ], dim=1).requires_grad_(True)
    
    def get_j_vals(pts_in):
        with torch.enable_grad():
            out = solver.model(pts_in)
            ones = torch.ones((pts_in.shape[0], 1), device=DEVICE)
            def vgrad(val): return grad(val, pts_in, ones, create_graph=True, retain_graph=True)[0]
            gu, gv, gw = vgrad(out[:, 0:1]), vgrad(out[:, 1:2]), vgrad(out[:, 2:3])
            F11, F12, F13 = 1.0+gu[:,0:1], gu[:,1:2], gu[:,2:3]
            F21, F22, F23 = gv[:,0:1], 1.0+gv[:,1:2], gv[:,2:3]
            F31, F32, F33 = gw[:,0:1], gw[:,1:2], 1.0+gw[:,2:3]
            J_raw = (F11*(F22*F33-F23*F32) - F12*(F21*F33-F23*F31) + F13*(F21*F32-F22*F31))
            J = torch.abs(J_raw) + 1e-6
            return J.detach().cpu().numpy().flatten()

    j_curve = get_j_vals(pts_curve)
    j_samp = get_j_vals(pts_samp)
    
    # 3. 绘图
    plt.figure(figsize=(10, 6), dpi=200)
    # 背景连续曲线
    plt.plot(r_norm_curve, j_curve, 'b-', lw=1.5, alpha=0.7, label='PINN 连续预测')
    # 前景离散采样点
    plt.plot(r_norm_samp, j_samp, 'ro', markersize=8, label='采样点 (dz=H/2)')
    
    # 标注数值
    for rn, jv in zip(r_norm_samp, j_samp):
        plt.annotate(f"{jv:.3f}", (rn, jv), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')
    
    # 竖线标注 NP/AF 界面
    plt.axvline(x=0.7, color='k', linestyle='--', alpha=0.5, label='NP/AF 界面 (r/a=0.7)')
    
    plt.xlabel("归一化半径 r/a"); plt.ylabel("体积变化比 J")
    plt.title(f"体积变化比 J 沿径向分布图 (z={z_fixed:.1f} mm) {suffix}", fontweight='bold')
    plt.xlim(-0.05, 1.05); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(save_dir, f"10_J_径向分布{suffix}.png"), bbox_inches='tight')
    plt.close()
    
    # 4. 导出 CSV
    csv_data = np.column_stack([r_norm_samp, x_samp, np.full_like(x_samp, z_fixed), j_samp])
    header = "r_norm,x_mm,z_mm,J_pinn"
    csv_dir = os.path.join(save_dir, "CSV_Data")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"j_radial_pinn{suffix}.csv")
    np.savetxt(csv_path, csv_data, delimiter=',', header=header, comments='', fmt='%.6f')
    print(f"✅ J 径向数据已导出: {csv_path}")

def generate_all_plots(solver, save_dir, suffix="", gen_deformed=True, gen_undeformed=True, gen_vtk=True):
    """
    统一可视化入口
    """
    generate_comsol_benchmark_figures(solver, save_dir, suffix=suffix,
                                      gen_deformed=gen_deformed, gen_undeformed=gen_undeformed, gen_vtk=gen_vtk)
    # 调用新增的 J 径向分布图功能
    plot_j_radial(solver, save_dir, suffix=suffix)
