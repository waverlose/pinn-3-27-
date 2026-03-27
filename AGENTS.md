# PINN IVD Poromechanics - Agent Guidelines

## Project Overview
Physics-Informed Neural Network (PINN) for Intervertebral Disc poromechanics modeling based on Triphasic Theory. The codebase implements a SIREN-based PINN for simulating mechanical and electrochemical behavior of vertebral discs.

## Build & Run Commands

### Core Execution
```bash
# Full training pipeline (train + potential visualization)
python main.py

# Post-processing and visualization (after training)
python run_post.py

# Export results for analysis
python export_results.py
```

### Training Control
- Training runs for `TRAIN_TOTAL_EPOCHS=20000` epochs by default
- Checkpoints saved every `SAVE_FREQ=500` epochs in `ivd_results/run_YYYYMMDD_HHMMSS/`
- Debug statistics printed every `DEBUG_STATS_FREQ=200` epochs
- RAS (Residual Adaptive Sampling) enabled with frequency `RAS_FREQ=100`

### Quick Testing
```bash
# Test solver initialization and basic functionality
python -c "import solver; s=solver.Solver(); print('OK')"

# Test with minimal data
python -c "
import torch, sys, os
sys.path.insert(0, '.')
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from solver import Solver
from geometry_material import GeometryMaterial
s=Solver()
gm=GeometryMaterial()
pts=gm.sample_domain(10).to('cuda' if torch.cuda.is_available() else 'cpu')
res=s.compute_physics(pts, True, True)
print(f'Test passed: {len(res)} results')
"
```

## Code Style & Conventions

### Imports Order
```python
# 1. Standard library
import os
import sys
import datetime

# 2. Third-party libraries  
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

# 3. Local modules (relative imports)
from config import *
from geometry_material import GeometryMaterial
from model import PINN
```

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `Solver`, `PINN`, `GeometryMaterial`)
- **Functions/Methods**: `snake_case` (e.g., `compute_physics`, `train_step`)
- **Variables**: `snake_case` (e.g., `target_disp_ratio`, `osmotic_ramp`)
- **Constants**: `UPPER_SNAKE_CASE` (in `config.py`)
- **Private methods**: `_leading_underscore` (e.g., `_memory_checkpoint`)

### Type Hints
- Use Python type hints for function signatures
- Prefer `torch.Tensor` over implicit types
- Use `from __future__ import annotations` for forward references

```python
def compute_physics(self, x_in: torch.Tensor, return_pde: bool = True, 
                   enable_chem: bool = True) -> tuple:
    """Compute physics residuals and related quantities."""
```

### Error Handling
- Use explicit exception handling for GPU operations
- Catch `RuntimeError` for CUDA/autograd issues
- Provide informative error messages in Chinese (primary) and English
- Clean up resources in `finally` blocks or context managers

```python
try:
    result = self.compute_physics(x, True, True)
except RuntimeError as e:
    # Log error, clean up, re-raise or handle gracefully
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    raise RuntimeError(f"Physics computation failed: {e}")
```

### Memory Management (CRITICAL)
- **GPU Memory**: Explicitly clean up with `torch.cuda.empty_cache()` every 10 epochs
- **Intermediate Tensors**: Delete explicitly after use (`del tensor`)
- **Compute Graph**: Avoid unnecessary `retain_graph=True`; clean up references
- **Device Consistency**: Ensure all tensors are on correct device (use `_device_check`)

```python
# Good practice:
res = self.compute_physics(pts, True, True)
loss = torch.mean(res[0]**2 + res[1]**2)
loss.backward()
del res  # Explicit cleanup
self._safe_cleanup(intermediate_tensors)
```

### Physics Implementation Guidelines
- Follow COMSOL `PINNmodel.m` reference implementation
- Maintain hard constraints (elliptic symmetry, boundary conditions)
- Use adaptive weights for loss balancing (`adaptive_weights` dict)
- Implement curriculum learning (ramping osmotic pressure, chemistry)
- Preserve numerical stability with clamping and epsilon values

### Configuration
- All parameters centralized in `config.py`
- Never hardcode values; use constants from config
- Device selection: `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Geometry: `GEO_A, GEO_B, GEO_H = 20.0, 15.0, 10.0`
- Material properties: `MAT_NP_MU, MAT_AF_MU`, etc.

### Testing & Validation
- No formal test suite; rely on runtime checks
- Use `debug_stats()` for training monitoring
- Validate against COMSOL results where available
- Check physical plausibility (J > 0, pressures reasonable)

### File Organization
- `main.py`: Entry point, dependency checks, training orchestration
- `solver.py`: Core physics implementation, training steps
- `train.py`: Training loop, curriculum scheduling, RAS sampling  
- `model.py`: SIREN neural network architecture
- `config.py`: All constants and hyperparameters
- `geometry_material.py`: Domain sampling and material properties
- `visualization.py`: Plotting and VTK export (post-training)
- `export_results.py`: Data extraction for analysis

### Git & Workflow
- Commit messages in Chinese for consistency
- Back up code snapshots automatically during training
- Results saved in timestamped directories: `ivd_results/run_YYYYMMDD_HHMMSS/`
- Include full code snapshot in result directories

## Common Issues & Solutions

### CUDA Memory Errors
1. **Access Violation (-1073741819)**: Increase memory cleanup frequency, reduce batch sizes if possible, ensure intermediate tensor cleanup
2. **Out of Memory**: Check `TRAIN_BATCH_DOM` (5000) and `TRAIN_BATCH_BC` (2000); consider reducing if GPU < 8GB VRAM
3. **Memory Fragmentation**: Call `torch.cuda.empty_cache()` regularly (every 10 epochs)

### Training Stability
1. **NaN/Inf Values**: Check clamping in `compute_physics` (e.g., `torch.clamp`, `+ 1e-12`)
2. **Exploding Gradients**: Use adaptive weights, gradient clipping (not currently implemented)
3. **Slow Convergence**: Adjust learning rate (`TRAIN_INIT_LR=2e-4`), curriculum parameters

### Performance Optimization
1. **RAS Sampling**: Occurs every 100 epochs; can be disabled via `RAS_ENABLED=False`
2. **Batch Size**: Primary memory consumer; modify only if necessary
3. **Mixed Precision**: Not implemented; could be future optimization

## Notes for AI Agents
- **Parameter Changes**: Avoid modifying `config.py` constants unless explicitly requested
- **Architecture Focus**: Optimize memory usage and stability through code structure, not parameter tuning
- **Chinese Comments**: Preserve existing Chinese comments; add English explanations if needed
- **COMSOL Alignment**: Maintain compatibility with reference MATLAB implementation
- **Physical Constraints**: Never break hard-coded physics constraints (elliptic symmetry, etc.)

## Dependencies
- Python 3.8+
- PyTorch (CUDA recommended)
- NumPy, Matplotlib
- tqdm for progress bars
- (Optional) pyevtk for VTK export