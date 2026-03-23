"""
训练脚本 - 高信息量汉化版
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import datetime
from tqdm import tqdm
import torch
import numpy as np
from solver import Solver
from config import *

def create_run_folder(base_dir=OUTPUT_BASE_DIR):
    import shutil
    import glob
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    save_path = os.path.join(base_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    
    # 备份当前代码快照
    code_backup_dir = os.path.join(save_path, "code_snapshot")
    os.makedirs(code_backup_dir, exist_ok=True)
    for py_file in glob.glob("*.py"):
        shutil.copy(py_file, code_backup_dir)
        
    print(f"输出目录 (及代码快照已备份): {save_path}")
    return save_path

def train_model_from(solver, save_dir, start_epoch):
    print(f"启动物理信息神经网络训练...")
    
    # 修正：使用更兼容的占位符 {n_fmt}/{total}
    bar_format = "{desc}: {percentage:3.0f}%|{bar}| [{n_fmt}/{total}] 耗时:{elapsed} < 剩余:{remaining}, {rate_fmt}"
    pbar = tqdm(range(start_epoch, TRAIN_TOTAL_EPOCHS), desc="训练进度", ncols=120, ascii=True, bar_format=bar_format)
    t_start = time.time()

    pts_cache = None
    ras_candidates = None

    for epoch in pbar:
        # 1. 课程学习调度 (大幅延长加载过程)
        osmotic_ramp, chem_ramp, barrier_ramp = 0.0, 0.01, 1.0
        target_disp_ratio = 0.0
        
        if epoch < 500:
            phase = "阶0:参考态"; chem_ramp = 0.01 + 0.09 * (epoch / 500.0)
        elif epoch < 3000:
            phase = "阶1a:力学预热"; t_p = (epoch - 500) / 7500.0
            target_disp_ratio = t_p; chem_ramp = 0.1 + 0.1 * t_p; osmotic_ramp = 0.0
        elif epoch < 8000:
            phase = "阶1b:协同加载"; t_p = (epoch - 500) / 7500.0
            target_disp_ratio = t_p; chem_ramp = 0.1 + 0.1 * t_p 
            osmotic_ramp = 0.5 * t_p 
        elif epoch < 12000:
            phase = "阶2:渗透增强"; t_p = (epoch - 8000) / 4000.0
            target_disp_ratio = 1.0; osmotic_ramp = 0.5 + 0.2 * t_p; chem_ramp = 0.2 + 0.8 * t_p
        elif epoch < 15000:
            phase = "阶3:全场耦合"; t_p = (epoch - 12000) / 3000.0
            target_disp_ratio = 1.0; osmotic_ramp = 0.7 + 0.3 * t_p; chem_ramp = 1.0
        else:
            phase = "阶4:稳态精修"; target_disp_ratio, osmotic_ramp, chem_ramp = 1.0, 1.0, 1.0

        solver.set_curriculum_params(osmotic_ramp=osmotic_ramp, chem_ramp=chem_ramp, barrier_ramp=barrier_ramp)
        pbar.set_description(f"训练进度({phase})")

        # 2. RAS 采样
        if RAS_ENABLED and epoch % RAS_FREQ == 0:
            if ras_candidates is None: 
                ras_candidates = solver.gm.sample_domain(RAS_NUM_CANDIDATES).to(DEVICE)
            else:
                new_c = solver.gm.sample_domain(RAS_NUM_CANDIDATES // 2).to(DEVICE)
                ras_candidates = torch.cat([ras_candidates[RAS_NUM_CANDIDATES // 2:], new_c], dim=0)
            
            res = solver.compute_residual_norm(ras_candidates, True).flatten()
            n_h = int(TRAIN_BATCH_DOM * RAS_MIX_RATIO)
            pts_h = ras_candidates[torch.topk(res, k=min(n_h, res.numel())).indices] if n_h > 0 else torch.empty((0,3), device=DEVICE)
            pts_r = solver.gm.sample_domain(TRAIN_BATCH_DOM - len(pts_h)).to(DEVICE)
            btm_pts, top_pts = solver.gm.sample_boundary_top_bottom(TRAIN_BATCH_BC)
            side_pts, side_norm = solver.gm.sample_boundary_side(TRAIN_BATCH_BC)
            pts_cache = {"dom": torch.cat([pts_h, pts_r], 0).detach(), "btm": btm_pts, "top": top_pts, "side": side_pts, "normals": side_norm}

            n_np_extra = 200
            theta_extra = torch.rand(n_np_extra, device=DEVICE) * 2 * np.pi
            r_extra = torch.sqrt(torch.rand(n_np_extra, device=DEVICE)) * NP_AXIS_RATIO * 0.3
            x_extra = r_extra * torch.cos(theta_extra) * GEO_A
            y_extra = r_extra * torch.sin(theta_extra) * GEO_B
            z_extra = torch.rand(n_np_extra, device=DEVICE) * GEO_H
            pts_np_extra = torch.stack([x_extra, y_extra, z_extra], dim=1)
            pts_cache["dom"] = torch.cat([pts_cache["dom"], pts_np_extra], dim=0).detach()

        if pts_cache is None:
            btm, top = solver.gm.sample_boundary_top_bottom(TRAIN_BATCH_BC)
            side, norm = solver.gm.sample_boundary_side(TRAIN_BATCH_BC)
            pts_cache = {"dom": solver.gm.sample_domain(TRAIN_BATCH_DOM).to(DEVICE), "btm": btm, "top": top, "side": side, "normals": norm}

            n_np_extra = 200
            theta_extra = torch.rand(n_np_extra, device=DEVICE) * 2 * np.pi
            r_extra = torch.sqrt(torch.rand(n_np_extra, device=DEVICE)) * NP_AXIS_RATIO * 0.3
            x_extra = r_extra * torch.cos(theta_extra) * GEO_A
            y_extra = r_extra * torch.sin(theta_extra) * GEO_B
            z_extra = torch.rand(n_np_extra, device=DEVICE) * GEO_H
            pts_np_extra = torch.stack([x_extra, y_extra, z_extra], dim=1)
            pts_cache["dom"] = torch.cat([pts_cache["dom"], pts_np_extra], dim=0).detach()

        # 3. 训练步
        # 修正：在协同加载开始前保持权重稳定
        uw = (epoch >= 8000 and epoch % 10 == 0)
        
        if epoch < 8000:
            solver.adaptive_weights["pde_mom"] = torch.tensor(1.0, device=DEVICE)
            solver.adaptive_weights["pde_flow"] = torch.tensor(0.01, device=DEVICE)
            solver.adaptive_weights["pde_ion"] = torch.tensor(0.01, device=DEVICE)

        loss_t, pde_val, bc_val = solver.train_step(TRAIN_BATCH_DOM, TRAIN_BATCH_BC, target_disp_ratio, pts_cache, True, update_weights=uw)

        # 记录简单的损失以便 tqdm 显示
        solver.scheduler.step()
        
        if epoch % DEBUG_STATS_FREQ == 0:
            # 这里的统计报表会包含 损失、残差、权重、J值
            solver.debug_stats(pts_cache["dom"][:1000], epoch, True, loss_t, pde_val, bc_val)

        if epoch % SAVE_FREQ == 0 and epoch > 0:
            solver.model.save(os.path.join(save_dir, f"ckpt_ep_{epoch}.pth"))
    
    print(f"\n🎯 进入 L-BFGS 二阶微调阶段...")
    for i in range(500):
        lbfgs_loss = solver.train_step_lbfgs(TRAIN_BATCH_DOM, TRAIN_BATCH_BC, 1.0, pts_cache, True)
        if i % 10 == 0: print(f"  L-BFGS [步数 {i}/500] 损失: {lbfgs_loss:.2e}")
    
    return time.time() - t_start

def main():
    save_dir = create_run_folder()
    solver = Solver()
    try:
        elapsed = train_model_from(solver, save_dir, 0)
        solver.model.save(os.path.join(save_dir, "model_final.pth"))
        print(f"✅ 训练成功结束! 总耗时: {elapsed/60:.1f} 分钟")
        return solver, save_dir 
    except KeyboardInterrupt:
        solver.model.save(os.path.join(save_dir, "model_stop.pth"))
        return solver, save_dir
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        solver.model.save(os.path.join(save_dir, "model_crash.pth"))
        return None, None

if __name__ == "__main__":
    main()
