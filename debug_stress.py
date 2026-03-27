import torch
import sys
import os
sys.path.insert(0, '.')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import importlib
import config
importlib.reload(config)
from config import DEVICE, BC_TOP_PRESSURE
from solver import Solver
from geometry_material import GeometryMaterial

solver = Solver()
gm = GeometryMaterial()
btm_pts, top_pts = gm.sample_boundary_top_bottom(100)
top_pts = top_pts.to(DEVICE)

with torch.no_grad():
    _, _, _, _, _, st, _, _, _, _, _, _, _ = solver.compute_physics(top_pts, False, True)
    s33 = st[2]
    print(f"s33 shape: {s33.shape}")
    print(f"s33 min: {s33.min().item():.3f}, max: {s33.max().item():.3f}, mean: {s33.mean().item():.3f}, std: {s33.std().item():.3f}")
    print(f"Pressure target: {BC_TOP_PRESSURE}")
    print(f"Loss per point: {(s33 + BC_TOP_PRESSURE)**2}")
    print(f"Total loss: {torch.mean((s33 + BC_TOP_PRESSURE)**2).item():.3f}")
    
    # Also compute displacement outputs
    out = solver.model(top_pts)
    print(f"Displacement u mean: {out[:,0].mean().item():.3e}, v: {out[:,1].mean().item():.3e}, w: {out[:,2].mean().item():.3e}")
    print(f"Displacement scale factor: {config.MODEL_OUT_SCALE_U}")
    # Compute raw_out
    # raw_out = out / scale? Actually displacement is scaled by MODEL_OUT_SCALE_U after elliptic symmetry.
    # Let's compute hat_u = out[:,0] / top_pts[:,0] (since u = x * hat_u)
    hat_u = out[:,0] / (top_pts[:,0] + 1e-12)
    hat_v = out[:,1] / (top_pts[:,1] + 1e-12)
    print(f"hat_u mean: {hat_u.mean().item():.3e}, hat_v mean: {hat_v.mean().item():.3e}")