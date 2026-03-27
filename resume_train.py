"""
断点续训脚本 (Resume Training Script)
用法:
    python resume_train.py --checkpoint <检查点路径> [--epoch <起始epoch>]
    
示例:
    python resume_train.py --checkpoint ivd_results/run_20260327_092505/ckpt_ep_6500.pth
    python resume_train.py --checkpoint ivd_results/run_20260327_092505/ckpt_ep_6500.pth --epoch 6500

如果不指定epoch，将自动从检查点文件名中提取epoch编号。
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import argparse
import time
import datetime
import logging
import threading
from tqdm import tqdm
import torch
import numpy as np
from solver import Solver
from config import *
from train import setup_logger, HeartbeatMonitor, create_run_folder

# ============================================================================
# 参数解析
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='PINN 断点续训')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='检查点文件路径 (.pth)')
    parser.add_argument('--epoch', '-e', type=int, default=None,
                       help='起始epoch (不指定则从文件名提取)')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help='输出目录 (不指定则继续使用原目录)')
    return parser.parse_args()

def extract_epoch_from_path(ckpt_path):
    """从检查点路径提取epoch编号"""
    import re
    filename = os.path.basename(ckpt_path)
    match = re.search(r'ckpt_ep_(\d+)\.pth', filename)
    if match:
        return int(match.group(1))
    match = re.search(r'model_(\w+)\.pth', filename)
    if match:
        # 对于 model_final.pth, model_best.pth 等，返回 TRAIN_TOTAL_EPOCHS
        return TRAIN_TOTAL_EPOCHS
    return 0

# ============================================================================
# 主函数
# ============================================================================
def main():
    args = parse_args()
    
    # 检查检查点文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"错误: 检查点文件不存在: {args.checkpoint}")
        sys.exit(1)
    
    # 确定起始epoch
    start_epoch = args.epoch if args.epoch is not None else extract_epoch_from_path(args.checkpoint)
    
    # 确定输出目录
    if args.output_dir:
        save_dir = args.output_dir
    else:
        # 使用原检查点所在目录
        save_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    
    print("="*70)
    print("断点续训模式")
    print("="*70)
    print(f"检查点: {args.checkpoint}")
    print(f"起始epoch: {start_epoch}")
    print(f"输出目录: {save_dir}")
    print("="*70)
    
    # 设置日志
    logger = setup_logger(save_dir)
    logger.info("="*70)
    logger.info("断点续训开始")
    logger.info(f"检查点: {args.checkpoint}")
    logger.info(f"起始epoch: {start_epoch}")
    logger.info("="*70)
    
    # 创建求解器并加载检查点
    solver = Solver()
    loaded_epoch = solver.load_checkpoint(args.checkpoint)
    logger.info(f"检查点加载成功，实际epoch: {loaded_epoch}")
    
    # 如果加载的epoch与指定的不同，使用加载的epoch
    if loaded_epoch > 0:
        start_epoch = loaded_epoch
    
    if start_epoch >= TRAIN_TOTAL_EPOCHS:
        logger.warning(f"起始epoch ({start_epoch}) 已达到总epochs ({TRAIN_TOTAL_EPOCHS})，无需继续训练")
        print("训练已完成，无需续训。")
        return
    
    # 启动心跳监控
    heartbeat = HeartbeatMonitor(logger, check_interval=30, max_silence=120)
    heartbeat.start()
    
    # 训练循环
    bar_format = "{desc}: {percentage:3.0f}%|{bar}| [{n_fmt}/{total}] 耗时:{elapsed} < 剩余:{remaining}, {rate_fmt}"
    pbar = tqdm(range(start_epoch, TRAIN_TOTAL_EPOCHS), desc="续训进度", ncols=120, ascii=True, bar_format=bar_format)
    t_start = time.time()
    
    pts_cache = None
    ras_candidates = None
    best_loss = float('inf')
    
    try:
        for epoch in pbar:
            heartbeat.update(epoch)
            
            # 课程学习调度 - 与 train.py 保持一致
            osmotic_ramp, chem_ramp, barrier_ramp = 0.0, 0.01, 1.0
            target_disp_ratio = 0.0
            
            if epoch < 2000:
                phase = "阶0:力学预压"
                t = epoch / 2000.0
                target_disp_ratio = 0.5 * t
                osmotic_ramp = 0.0
                chem_ramp = 0.01
                
            elif epoch < 5000:
                phase = "阶1:力学加载"
                t = (epoch - 2000) / 3000.0
                target_disp_ratio = 0.5 + 0.5 * t
                osmotic_ramp = 0.0
                chem_ramp = 0.01
                
            elif epoch < 8000:
                phase = "阶2:渗透压引入"
                target_disp_ratio = 1.0
                t = (epoch - 5000) / 3000.0
                osmotic_ramp = 1.0 * t
                chem_ramp = 0.01 + 0.99 * t
                
            elif epoch < 12000:
                phase = "阶3:全耦合"
                target_disp_ratio = 1.0
                osmotic_ramp = 1.0
                chem_ramp = 1.0
                
            else:
                phase = "阶4:稳态优化"
                target_disp_ratio = 1.0
                osmotic_ramp = 1.0
                chem_ramp = 1.0
            
            solver.set_curriculum_params(osmotic_ramp=osmotic_ramp, chem_ramp=chem_ramp, barrier_ramp=barrier_ramp)
            pbar.set_description(f"续训进度({phase})")
            
            # RAS采样
            if RAS_ENABLED and epoch % RAS_FREQ == 0:
                if ras_candidates is None:
                    ras_candidates = solver.gm.sample_domain(RAS_NUM_CANDIDATES).to(DEVICE)
                else:
                    new_c = solver.gm.sample_domain(RAS_NUM_CANDIDATES // 2).to(DEVICE)
                    old_ras = ras_candidates
                    ras_candidates = torch.cat([old_ras[RAS_NUM_CANDIDATES // 2:], new_c], dim=0)
                    del old_ras

                res = solver.compute_residual_norm(ras_candidates, True).flatten()
                n_h = int(TRAIN_BATCH_DOM * RAS_MIX_RATIO)
                pts_h = ras_candidates[torch.topk(res, k=min(n_h, res.numel())).indices] if n_h > 0 else torch.empty((0,3), device=DEVICE)
                pts_r = solver.gm.sample_domain(TRAIN_BATCH_DOM - len(pts_h)).to(DEVICE)
                btm_pts, top_pts = solver.gm.sample_boundary_top_bottom(TRAIN_BATCH_BC)
                side_pts, side_norm = solver.gm.sample_boundary_side(TRAIN_BATCH_BC)
                btm_pts, top_pts = btm_pts.to(DEVICE), top_pts.to(DEVICE)
                side_pts, side_norm = side_pts.to(DEVICE), side_norm.to(DEVICE)

                if pts_cache is not None:
                    for key in list(pts_cache.keys()):
                        del pts_cache[key]
                    pts_cache = None

                pts_cache = {"dom": torch.cat([pts_h, pts_r], 0).detach(), "btm": btm_pts, "top": top_pts, "side": side_pts, "normals": side_norm}

                n_np_extra = 200
                theta_extra = torch.rand(n_np_extra, device=DEVICE) * 2 * np.pi
                r_extra = torch.sqrt(torch.rand(n_np_extra, device=DEVICE)) * NP_AXIS_RATIO * 0.3
                x_extra = r_extra * torch.cos(theta_extra) * GEO_A
                y_extra = r_extra * torch.sin(theta_extra) * GEO_B
                z_extra = torch.rand(n_np_extra, device=DEVICE) * GEO_H
                pts_np_extra = torch.stack([x_extra, y_extra, z_extra], dim=1)
                pts_cache["dom"] = torch.cat([pts_cache["dom"], pts_np_extra], dim=0).detach()

                del res, pts_h, pts_r, btm_pts, top_pts, side_pts, side_norm, pts_np_extra
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if pts_cache is None:
                btm, top = solver.gm.sample_boundary_top_bottom(TRAIN_BATCH_BC)
                side, norm = solver.gm.sample_boundary_side(TRAIN_BATCH_BC)
                btm, top = btm.to(DEVICE), top.to(DEVICE)
                side, norm = side.to(DEVICE), norm.to(DEVICE)

                pts_cache = {"dom": solver.gm.sample_domain(TRAIN_BATCH_DOM).to(DEVICE), "btm": btm, "top": top, "side": side, "normals": norm}

                n_np_extra = 200
                theta_extra = torch.rand(n_np_extra, device=DEVICE) * 2 * np.pi
                r_extra = torch.sqrt(torch.rand(n_np_extra, device=DEVICE)) * NP_AXIS_RATIO * 0.3
                x_extra = r_extra * torch.cos(theta_extra) * GEO_A
                y_extra = r_extra * torch.sin(theta_extra) * GEO_B
                z_extra = torch.rand(n_np_extra, device=DEVICE) * GEO_H
                pts_np_extra = torch.stack([x_extra, y_extra, z_extra], dim=1)
                pts_cache["dom"] = torch.cat([pts_cache["dom"], pts_np_extra], dim=0).detach()
            
            # 训练步
            solver.set_curriculum_params(osmotic_ramp=osmotic_ramp, chem_ramp=chem_ramp, barrier_ramp=barrier_ramp)
            uw = (epoch >= 8000 and epoch % 10 == 0)
            
            loss_t, pde_val, bc_val = solver.train_step(
                TRAIN_BATCH_DOM, TRAIN_BATCH_BC, target_disp_ratio, pts_cache, True, update_weights=uw, epoch=epoch
            )
            
            if torch.cuda.is_available() and epoch % 10 == 0:
                torch.cuda.empty_cache()
            
            solver.scheduler.step()
            
            # 日志和检查点
            if epoch % DEBUG_STATS_FREQ == 0:
                solver.debug_stats(pts_cache["dom"][:1000], epoch, True, loss_t, pde_val, bc_val)
                logger.info(f"Epoch {epoch}: loss={loss_t:.4e}, pde={pde_val:.4e}, bc={bc_val:.4e}")
            
            if epoch % SAVE_FREQ == 0 and epoch > 0:
                ckpt_path = os.path.join(save_dir, f"ckpt_ep_{epoch}.pth")
                solver.save_checkpoint(ckpt_path, epoch)
                logger.info(f"检查点已保存: {ckpt_path}")
                
                if loss_t < best_loss:
                    best_loss = loss_t
                    best_path = os.path.join(save_dir, "model_best.pth")
                    solver.save_checkpoint(best_path, epoch)
                    
        # L-BFGS优化
        logger.info("开始L-BFGS二阶优化...")
        print(f"\nL-BFGS 二阶优化...")
        for i in range(100):
            lbfgs_loss = solver.train_step_lbfgs(TRAIN_BATCH_DOM, TRAIN_BATCH_BC, 1.0, pts_cache, True)
            if i % 10 == 0:
                logger.info(f"L-BFGS step {i}: loss={lbfgs_loss:.4e}")
                print(f" L-BFGS [步数 {i}/100] 损失: {lbfgs_loss:.2e}")
        
        # 保存最终模型
        elapsed = time.time() - t_start
        final_path = os.path.join(save_dir, "model_final.pth")
        solver.save_checkpoint(final_path, TRAIN_TOTAL_EPOCHS)
        
        heartbeat.stop()
        
        logger.info("="*70)
        logger.info(f"续训成功结束! 本次耗时: {elapsed/60:.1f} 分钟")
        logger.info("="*70)
        print(f"\n✅ 续训成功结束! 本次耗时: {elapsed/60:.1f} 分钟")
        
    except KeyboardInterrupt:
        logger.warning("用户中断续训")
        stop_path = os.path.join(save_dir, f"ckpt_stop_{start_epoch}.pth")
        solver.save_checkpoint(stop_path, start_epoch)
        heartbeat.stop()
        
    except Exception as e:
        logger.error(f"续训失败: {e}")
        import traceback
        traceback.print_exc()
        error_path = os.path.join(save_dir, f"ckpt_error_{start_epoch}.pth")
        solver.save_checkpoint(error_path, start_epoch)
        heartbeat.stop()

if __name__ == "__main__":
    main()
