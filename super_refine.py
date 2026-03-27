import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from solver import Solver
from geometry_material import GeometryMaterial
from config import *

class ModelRefiner:
    def __init__(self, run_dir: str, checkpoint_name: str):
        self.run_dir = run_dir
        self.checkpoint_path = os.path.join(run_dir, checkpoint_name)
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
        self.gm = GeometryMaterial()
        self.solver = Solver()
        self.device = DEVICE

    def _generate_points(self):
        side_pts, side_norms = self.gm.sample_boundary_side(TRAIN_BATCH_BC)
        btm_pts, top_pts = self.gm.sample_boundary_top_bottom(TRAIN_BATCH_BC * 2)
        return {
            "dom": self.gm.sample_domain(TRAIN_BATCH_DOM).to(self.device),
            "btm": btm_pts.to(self.device),
            "top": top_pts.to(self.device),
            "side": side_pts.to(self.device),
            "normals": side_norms.to(self.device)
        }

    def execute(self, adam_steps: int = 500, lbfgs_steps: int = 1000):
        try:
            torch.cuda.empty_cache()
            self.solver.model.load(self.checkpoint_path)
            
            self.solver.set_curriculum_params(osmotic_ramp=1.0, chem_ramp=1.0, barrier_ramp=1.0)
            self.solver.adaptive_weights["trend"] = torch.tensor(30.0, device=self.device)

            pts_cache = self._generate_points()
            
            pbar = tqdm(total=adam_steps, desc="Adam Stage")
            for i in range(adam_steps):
                update_w = (i % 20 == 0)
                loss, _, _ = self.solver.train_step(
                    n_dom=TRAIN_BATCH_DOM,
                    n_bc=TRAIN_BATCH_BC,
                    target_disp_ratio=1.0,
                    pts=pts_cache,
                    enable_chem=True,
                    update_weights=update_w
                )
                
                pbar.set_postfix({
                    "loss": f"{loss:.2e}",
                    "trend_w": f"{self.solver.adaptive_weights['trend'].item():.1f}"
                })
                pbar.update(1)
            pbar.close()

            print("Starting L-BFGS stage...")
            final_lbfgs_loss = self.solver.train_step_lbfgs(
                n_dom=TRAIN_BATCH_DOM,
                n_bc=TRAIN_BATCH_BC,
                target_disp_ratio=1.0,
                pts=pts_cache,
                enable_chem=True
            )
            
            output_path = os.path.join(self.run_dir, "model_polished.pth")
            self.solver.model.save(output_path)
            print(f"Refinement complete. Saved: {output_path} | L-BFGS Loss: {final_lbfgs_loss:.2e}")

        except Exception as e:
            print(f"Error during refinement: {e}")
            sys.exit(1)

if __name__ == "__main__":
    TARGET_DIR = r"ivd_results\run_20260323_154859"
    refiner = ModelRefiner(TARGET_DIR, "model_crash.pth")
    refiner.execute()
