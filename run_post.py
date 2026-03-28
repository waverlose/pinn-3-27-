"""
后处理脚本 - 仅用于生成图表 (Post-Processing Only)
无需重新训练，直接加载已有模型并导出对标云图。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import glob
from solver import Solver
from visualization import generate_all_plots
from export_results import export_results
from config import *

def main():
    runs = [d for d in os.listdir("ivd_results") if os.path.isdir(os.path.join("ivd_results", d))]
    if not runs:
        print("未找到运行记录。")
        return
    latest_run = os.path.join("ivd_results", sorted(runs)[-1])
    
    # 优先加载精修后的模型
    polished_path = os.path.join(latest_run, "model_polished.pth")
    final_path = os.path.join(latest_run, "model_final.pth")
    model_path = polished_path if os.path.exists(polished_path) else final_path
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Loading Model: {model_path}")
    solver = Solver()
    try:
        # 使用 Solver 内置的加载方法，它能正确解包复合检查点
        start_epoch = solver.load_checkpoint(model_path)
        print(f"Model weights loaded successfully (from epoch {start_epoch}).")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    post_dir = os.path.join(latest_run, f"Post_Analysis_{timestamp}")
    os.makedirs(post_dir, exist_ok=True)
    
    print(f"Generating benchmark figures to: {post_dir}")
    # 1. 生成变形/未变形云图
    generate_all_plots(solver, post_dir)
    
    # 2. 导出所有 CSV 数据到专属文件夹
    csv_dir = os.path.join(post_dir, "CSV_Data")
    os.makedirs(csv_dir, exist_ok=True)
    print(f"Exporting CSV data to: {csv_dir}")
    export_results(model_path, save_dir=csv_dir)
    
    print(f"\nPost-processing completed. Results saved in: {post_dir}")

if __name__ == "__main__":
    main()
