import torch
import sys
import os
sys.path.insert(0, '.')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import importlib
import config
importlib.reload(config)
from config import *
from solver import Solver
from geometry_material import GeometryMaterial
import time

print("Starting short training test with 20MPa pressure")
print(f"MODEL_OUT_SCALE_U = {MODEL_OUT_SCALE_U}")
print(f"BC_TOP_PRESSURE = {BC_TOP_PRESSURE}")
print(f"WEIGHT_STRATEGY = {WEIGHT_STRATEGY}")

solver = Solver()
gm = GeometryMaterial()

# Sample initial points
pts_cache = None
ras_candidates = None

total_epochs = 200  # short test
for epoch in range(total_epochs):
    # Curriculum schedule (simplified)
    if epoch < 500:
        target_disp_ratio = 0.0
        chem_ramp = 0.01 + 0.09 * (epoch / 500.0)
        osmotic_ramp = 0.0
    elif epoch < 3000:
        t_p = (epoch - 500) / 7500.0
        target_disp_ratio = t_p
        chem_ramp = 0.1 + 0.1 * t_p
        osmotic_ramp = 0.0
    else:
        target_disp_ratio = 1.0
        chem_ramp = 1.0
        osmotic_ramp = 1.0
    barrier_ramp = 1.0
    
    solver.set_curriculum_params(osmotic_ramp=osmotic_ramp, chem_ramp=chem_ramp, barrier_ramp=barrier_ramp)
    
    # RAS sampling disabled for speed
    if pts_cache is None:
        btm_pts, top_pts = gm.sample_boundary_top_bottom(TRAIN_BATCH_BC)
        side_pts, side_norm = gm.sample_boundary_side(TRAIN_BATCH_BC)
        dom_pts = gm.sample_domain(TRAIN_BATCH_DOM)
        pts_cache = {
            "dom": dom_pts.to(DEVICE),
            "btm": btm_pts.to(DEVICE),
            "top": top_pts.to(DEVICE),
            "side": side_pts.to(DEVICE),
            "normals": side_norm.to(DEVICE)
        }
    
    update_weights = (epoch % WEIGHT_UPDATE_FREQ == 0)
    
    loss_total, loss_pde, loss_bc = solver.train_step(
        TRAIN_BATCH_DOM, TRAIN_BATCH_BC, target_disp_ratio, pts_cache, True, update_weights, epoch
    )
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: total={loss_total:.2e}, pde={loss_pde:.2e}, bc={loss_bc:.2e}, target_ratio={target_disp_ratio:.3f}")
        # print weights
        w = solver.weight_manager
        print(f"  Weights: mom={w['pde_mom'].item():.2f}, flow={w['pde_flow'].item():.2f}, ion={w['pde_ion'].item():.2f}, bc={w['bc'].item():.2f}")
    
    # Clean memory every 50 epochs
    if epoch % 50 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()

print("Training test completed.")