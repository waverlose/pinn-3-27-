"""
训练脚本 - 改进版
1. 分阶段顺序加载策略（物理最优）
2. 异常捕获和详细日志
3. 心跳检测机制
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import datetime
import logging
import threading
from tqdm import tqdm
import torch
import numpy as np
from solver import Solver
from config import *

# ============================================================================
# 日志配置
# ============================================================================
def setup_logger(save_dir):
    """配置训练日志"""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清除已有处理器
    
    # 文件处理器
    fh = logging.FileHandler(os.path.join(save_dir, "train.log"), encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

# ============================================================================
# 心跳检测机制
# ============================================================================
class HeartbeatMonitor:
    """心跳检测，防止训练意外中断"""
    def __init__(self, logger, check_interval=30, max_silence=120):
        self.logger = logger
        self.check_interval = check_interval
        self.max_silence = max_silence
        self.last_update = time.time()
        self.running = True
        self.thread = None
        self.last_epoch = -1
        
    def update(self, epoch):
        """更新心跳"""
        self.last_update = time.time()
        self.last_epoch = epoch
        
    def _monitor(self):
        """后台监控线程"""
        while self.running:
            silence_time = time.time() - self.last_update
            if silence_time > self.max_silence:
                self.logger.warning(f"训练已静默 {silence_time:.0f} 秒 (最后epoch: {self.last_epoch})，可能已中断")
            time.sleep(self.check_interval)
            
    def start(self):
        """启动监控"""
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        
    def stop(self):
        """停止监控"""
        self.running = False

# ============================================================================
# 辅助函数
# ============================================================================
def create_run_folder(base_dir=OUTPUT_BASE_DIR):
    """创建运行结果文件夹"""
    import shutil
    import glob
    if not os.path.exists(base_dir): 
        os.makedirs(base_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    save_path = os.path.join(base_dir, run_name)
    os.makedirs(save_path, exist_ok=True)

    # 备份当前代码快照
    code_backup_dir = os.path.join(save_path, "code_snapshot")
    os.makedirs(code_backup_dir, exist_ok=True)
    for py_file in glob.glob("*.py"):
        shutil.copy(py_file, code_backup_dir)

    print(f"输出目录: {save_path}")
    return save_path

# ============================================================================
# 训练主循环
# ============================================================================
def train_model_from(solver, save_dir, start_epoch, logger):
    """从指定epoch开始训练"""
    logger.info(f"训练开始: start_epoch={start_epoch}, total_epochs={TRAIN_TOTAL_EPOCHS}")
    
    # 启动心跳监控
    heartbeat = HeartbeatMonitor(logger, check_interval=30, max_silence=120)
    heartbeat.start()
    
    bar_format = "{desc}: {percentage:3.0f}%|{bar}| [{n_fmt}/{total}] 耗时:{elapsed} < 剩余:{remaining}, {rate_fmt}"
    pbar = tqdm(range(start_epoch, TRAIN_TOTAL_EPOCHS), desc="训练进度", ncols=120, ascii=True, bar_format=bar_format)
    t_start = time.time()

    pts_cache = None
    ras_candidates = None
    best_loss = float('inf')

    for epoch in pbar:
        try:
            # 更新心跳
            heartbeat.update(epoch)
            
            # ====================================================================
            # 1. 课程学习调度 - 分阶段顺序加载（物理最优策略）
            # ====================================================================
            # 阶段0 (0-2000): 纯力学预压，渗透压关闭
            # 阶段1 (2000-5000): 力学满载，渗透压关闭
            # 阶段2 (5000-8000): 力学稳定后，引入渗透压
            # 阶段3 (8000-12000): 全耦合稳态
            # 阶段4 (12000+): 精细优化
            
            osmotic_ramp, chem_ramp, barrier_ramp = 0.0, 0.01, 1.0
            target_disp_ratio = 0.0
            
            if epoch < 2000:
                # 阶段0: 力学预压 (0 → 0.15 MPa, 即 target_disp_ratio: 0 → 0.5)
                phase = "阶0:力学预压"
                t = epoch / 2000.0
                target_disp_ratio = 0.5 * t  # 0 → 0.5
                osmotic_ramp = 0.0  # 渗透压关闭
                chem_ramp = 0.01    # 最小电化学权重
                
            elif epoch < 5000:
                # 阶段1: 力学满载 (0.15 → 0.3 MPa)
                phase = "阶1:力学加载"
                t = (epoch - 2000) / 3000.0
                target_disp_ratio = 0.5 + 0.5 * t  # 0.5 → 1.0
                osmotic_ramp = 0.0  # 渗透压继续关闭
                chem_ramp = 0.01
                
            elif epoch < 8000:
                # 阶段2: 引入渗透压（力学已稳定）
                phase = "阶2:渗透压引入"
                target_disp_ratio = 1.0  # 力学满载保持
                t = (epoch - 5000) / 3000.0
                osmotic_ramp = 1.0 * t   # 0 → 1.0
                chem_ramp = 0.01 + 0.99 * t  # 0.01 → 1.0
                
            elif epoch < 12000:
                # 阶段3: 全耦合稳态
                phase = "阶3:全耦合"
                target_disp_ratio = 1.0
                osmotic_ramp = 1.0
                chem_ramp = 1.0
                
            else:
                # 阶段4: 精细优化
                phase = "阶4:稳态优化"
                target_disp_ratio = 1.0
                osmotic_ramp = 1.0
                chem_ramp = 1.0

            solver.set_curriculum_params(osmotic_ramp=osmotic_ramp, chem_ramp=chem_ramp, barrier_ramp=barrier_ramp)
            pbar.set_description(f"训练进度({phase})")

            # ====================================================================
            # 2. RAS 采样
            # ====================================================================
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

                # NP区域额外采样
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

            # ====================================================================
            # 3. 训练步
            # ====================================================================
            solver.set_curriculum_params(osmotic_ramp=osmotic_ramp, chem_ramp=chem_ramp, barrier_ramp=barrier_ramp)

            # 权重更新：在全耦合阶段开始按频率更新
            uw = (epoch >= 8000 and epoch % 10 == 0)

            # 调用重构后的 train_step (内部已处理权重更新)
            loss_t, pde_val, bc_val = solver.train_step(
                TRAIN_BATCH_DOM, TRAIN_BATCH_BC, target_disp_ratio, pts_cache, True, update_weights=uw, epoch=epoch
            )

            # 定期清理GPU内存
            if torch.cuda.is_available() and epoch % 10 == 0:
                torch.cuda.empty_cache()

            solver.scheduler.step()

            # ====================================================================
            # 4. 日志和检查点
            # ====================================================================
            if epoch % DEBUG_STATS_FREQ == 0:
                solver.debug_stats(pts_cache["dom"][:1000], epoch, True, loss_t, pde_val, bc_val)
                # 记录到日志文件
                logger.info(f"Epoch {epoch}: loss={loss_t:.4e}, pde={pde_val:.4e}, bc={bc_val:.4e}, "
                           f"target_ratio={target_disp_ratio:.3f}, osmotic={osmotic_ramp:.3f}, chem={chem_ramp:.3f}")

            if epoch % SAVE_FREQ == 0 and epoch > 0:
                ckpt_path = os.path.join(save_dir, f"ckpt_ep_{epoch}.pth")
                solver.save_checkpoint(ckpt_path, epoch)
                logger.info(f"检查点已保存: {ckpt_path}")
                
                # 保存最佳模型
                if loss_t < best_loss:
                    best_loss = loss_t
                    best_path = os.path.join(save_dir, "model_best.pth")
                    solver.save_checkpoint(best_path, epoch)
                    logger.info(f"最佳模型已更新: loss={best_loss:.4e}")

        except RuntimeError as e:
            # CUDA错误处理
            logger.error(f"CUDA RuntimeError at epoch {epoch}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 保存错误检查点
            error_path = os.path.join(save_dir, f"ckpt_error_{epoch}.pth")
            solver.save_checkpoint(error_path, epoch)
            logger.info(f"错误检查点已保存: {error_path}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error at epoch {epoch}: {e}")
            error_path = os.path.join(save_dir, f"ckpt_error_{epoch}.pth")
            solver.save_checkpoint(error_path, epoch)
            raise

    # ========================================================================
    # 5. L-BFGS 二阶优化（仅在训练结束后）
    # ========================================================================
    logger.info("开始L-BFGS二阶优化...")
    print(f"\nL-BFGS 二阶优化...")
    for i in range(500):
        # 显式更新心跳，防止误报
        heartbeat.update(15000 + i)
        lbfgs_loss = solver.train_step_lbfgs(TRAIN_BATCH_DOM, TRAIN_BATCH_BC, 1.0, pts_cache, True)
        if i % 10 == 0:
            logger.info(f"L-BFGS step {i}: loss={lbfgs_loss:.4e}")
            print(f" L-BFGS [步数 {i}/500] 损失: {lbfgs_loss:.2e}")

    # 停止心跳监控
    heartbeat.stop()
    
    return time.time() - t_start

# ============================================================================
# 主函数
# ============================================================================
def main():
    """主训练入口"""
    save_dir = create_run_folder()
    logger = setup_logger(save_dir)
    solver = Solver()
    
    try:
        logger.info("="*70)
        logger.info("训练开始")
        logger.info(f"设备: {DEVICE}")
        logger.info(f"总epochs: {TRAIN_TOTAL_EPOCHS}")
        logger.info("="*70)
        
        elapsed = train_model_from(solver, save_dir, 0, logger)
        
        # 保存最终模型
        final_path = os.path.join(save_dir, "model_final.pth")
        solver.save_checkpoint(final_path, TRAIN_TOTAL_EPOCHS)
        
        logger.info("="*70)
        logger.info(f"训练成功结束! 总耗时: {elapsed/60:.1f} 分钟")
        logger.info("="*70)
        print(f"\n✅ 训练成功结束! 总耗时: {elapsed/60:.1f} 分钟")
        
        return solver, save_dir
        
    except KeyboardInterrupt:
        logger.warning("用户中断训练")
        stop_path = os.path.join(save_dir, "model_stop.pth")
        solver.save_checkpoint(stop_path, 0)
        return solver, save_dir
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        crash_path = os.path.join(save_dir, "model_crash.pth")
        solver.save_checkpoint(crash_path, 0)
        return None, None

if __name__ == "__main__":
    main()
