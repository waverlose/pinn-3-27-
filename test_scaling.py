import torch
import sys
import os
sys.path.insert(0, '.')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Reload config to get updated value
import importlib
import config
importlib.reload(config)
from config import DEVICE, TRAIN_BATCH_DOM, TRAIN_BATCH_BC, BC_TOP_PRESSURE, MODEL_OUT_SCALE_U

from solver import Solver
from geometry_material import GeometryMaterial

print(f"Testing with MODEL_OUT_SCALE_U = {MODEL_OUT_SCALE_U}")
print(f"BC_TOP_PRESSURE = {BC_TOP_PRESSURE} MPa")

solver = Solver()
gm = GeometryMaterial()

# Sample points
dom_pts = gm.sample_domain(TRAIN_BATCH_DOM).to(DEVICE)
btm_pts, top_pts = gm.sample_boundary_top_bottom(TRAIN_BATCH_BC)
btm_pts = btm_pts.to(DEVICE)
top_pts = top_pts.to(DEVICE)
side_pts, side_norm = gm.sample_boundary_side(TRAIN_BATCH_BC)
side_pts = side_pts.to(DEVICE)
side_norm = side_norm.to(DEVICE)

pts = {"dom": dom_pts, "btm": btm_pts, "top": top_pts, "side": side_pts, "normals": side_norm}

# Get initial boundary loss components
with torch.no_grad():
    # Bottom displacement
    out_b = solver.model(pts["btm"])
    loss_bottom = torch.mean(out_b[:, 0:3]**2).item()
    
    # Top displacement (horizontal)
    out_t = solver.model(pts["top"])
    loss_top_horiz = torch.mean(out_t[:, 0:2]**2).item()
    
    # Top pressure boundary with target_disp_ratio = 1
    _, _, _, _, _, st, _, _, _, _, _, _, _ = solver.compute_physics(pts["top"], False, True)
    s33 = st[2]
    loss_pressure = torch.mean((s33 + 1.0 * BC_TOP_PRESSURE)**2).item()
    
    # Side boundary
    _, _, _, _, _, s_side, _, _, _, _, _, _, _ = solver.compute_physics(pts["side"], False, True)
    tx = s_side[0]*pts["normals"][:,0:1] + s_side[3]*pts["normals"][:,1:2]
    ty = s_side[3]*pts["normals"][:,0:1] + s_side[1]*pts["normals"][:,1:2]
    tz = s_side[4]*pts["normals"][:,0:1] + s_side[5]*pts["normals"][:,1:2]
    loss_side = torch.mean(tx**2+ty**2+tz**2).item()
    
    # Chemical boundary (simplified)
    loss_chem = torch.mean(solver.model(pts["side"])[:, 3:5]**2).item()
    
    # Total boundary loss (without weights)
    total = loss_bottom + loss_top_horiz + loss_pressure + loss_side + loss_chem
    
    print(f"Loss bottom disp: {loss_bottom:.2e}")
    print(f"Loss top horiz disp: {loss_top_horiz:.2e}")
    print(f"Loss pressure (target full): {loss_pressure:.2e}")
    print(f"Loss side traction: {loss_side:.2e}")
    print(f"Loss chemical: {loss_chem:.2e}")
    print(f"Total boundary loss (unweighted): {total:.2e}")
    
    # Also compute PDE residuals
    res = solver.compute_physics(dom_pts, True, True)
    pde_mom = torch.mean(res[0]**2 + res[1]**2 + res[2]**2).item()
    pde_flow = torch.mean(res[3]**2).item()
    pde_ion = torch.mean(res[4]**2).item()
    print(f"PDE momentum residual: {pde_mom:.2e}")
    print(f"PDE flow residual: {pde_flow:.2e}")
    print(f"PDE ion residual: {pde_ion:.2e}")
    
    # Weight values
    w = solver.weight_manager
    print(f"Weights: pde_mom={w['pde_mom'].item():.2f}, pde_flow={w['pde_flow'].item():.2f}, pde_ion={w['pde_ion'].item():.2f}, bc={w['bc'].item():.2f}")
    
    # Compute J values
    J = res[12]
    print(f"J min={J.min().item():.3f}, max={J.max().item():.3f}, mean={J.mean().item():.3f}")

print("Test completed.")