"""
主运行脚本
Main Execution Script - 完整流程：训练 + 可视化
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import numpy as np
import warnings

# 屏蔽无意义的显存分配器警告
warnings.filterwarnings("ignore", category=UserWarning, message="expandable_segments not supported")
from config import *
from train import main as train_main
from visualization import generate_all_plots


def check_dependencies():
    """检查依赖库是否安装"""
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(name)
    
    if missing:
        print("[ERROR] Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nPlease install them using:")
        print("   pip install torch numpy matplotlib tqdm")
        return False
    return True


def print_header():
    """打印程序头部信息"""
    print("=" * 80)
    print("Physics-Informed Neural Network for IVD Poromechanics")
    print("   Intervertebral Disc Hyperelasticity Physics-Informed Neural Network")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80 + "\n")


def main():
    """主函数：执行完整流程"""
    if not check_dependencies(): return
    print_header()
    
    # 设置随机种子
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    try:
        # 步骤1: 训练模型
        print("Step 1: Training Model")
        print("-" * 80)
        res = train_main()
        if res is None or res[0] is None:
            print("\nXXX Training failed or was cancelled.")
            sys.exit(1)
        solver, save_dir = res
        
        # 步骤2: 生成可视化 (已按要求关闭，请手动运行 run_post.py)
        # print("\nStep 2: Generating Visualizations")
        # print("-" * 80)
        # generate_all_plots(solver, save_dir)
        
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print(f"Results are saved in: {save_dir}")
        print("=" * 80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
