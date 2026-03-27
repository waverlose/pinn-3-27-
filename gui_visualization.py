"""
IVD-PINN 后处理可视化图形工具 (GUI Visualization Tool) - 修复版
修复了在某些 Windows/Anaconda 环境下点击浏览按钮崩溃的问题。
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import torch
import threading

# 导入本项目组件
try:
    from solver import Solver
    from visualization import generate_all_plots
    from config import DEVICE
except ImportError:
    print("错误：请确保在项目根目录下运行此脚本")
    sys.exit(1)

class VisualizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IVD-PINN ")
        self.root.geometry("700x520")
        
        # 尝试修复 DPI 缩放问题
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass
            
        self.selected_dir = tk.StringVar()
        self.model_files = []
        self.selected_model = tk.StringVar()
        
        self.setup_ui()

    def setup_ui(self):
        # 1. 文件夹选择
        frame_dir = ttk.LabelFrame(self.root, text=" 1. 选择运行结果文件夹 ", padding=10)
        frame_dir.pack(fill="x", padx=20, pady=10)
        
        self.entry_dir = ttk.Entry(frame_dir, textvariable=self.selected_dir, width=60)
        self.entry_dir.pack(side="left", padx=5)
        ttk.Button(frame_dir, text="浏览...", command=self.browse_folder).pack(side="left")

        # 2. 模型选择
        frame_model = ttk.LabelFrame(self.root, text=" 2. 选择权重文件 (.pth) ", padding=10)
        frame_model.pack(fill="x", padx=20, pady=10)
        
        self.model_combo = ttk.Combobox(frame_model, textvariable=self.selected_model, width=70, state="readonly")
        self.model_combo.pack(fill="x", padx=5)

        # 3. 后处理选项 (合三为一核心区)
        frame_opt = ttk.LabelFrame(self.root, text=" 3. 内容选择 ", padding=10)
        frame_opt.pack(fill="x", padx=20, pady=10)
        
        # 选项变量
        self.var_deformed = tk.BooleanVar(value=True)
        self.var_undeformed = tk.BooleanVar(value=True)
        self.var_csv = tk.BooleanVar(value=True)
        self.var_vtk = tk.BooleanVar(value=False)
        self.suffix_var = tk.StringVar(value="")
        
        # 第一排：复选框
        opts_frame1 = ttk.Frame(frame_opt)
        opts_frame1.pack(fill="x", pady=2)
        ttk.Checkbutton(opts_frame1, text="变形云图 (Deformed)", variable=self.var_deformed).pack(side="left", padx=10)
        ttk.Checkbutton(opts_frame1, text="未变形云图 (Undeformed)", variable=self.var_undeformed).pack(side="left", padx=10)
        ttk.Checkbutton(opts_frame1, text="导出全量 CSV", variable=self.var_csv).pack(side="left", padx=10)
        ttk.Checkbutton(opts_frame1, text="导出 3D VTK", variable=self.var_vtk).pack(side="left", padx=10)
        
        # 第二排：参数
        opts_frame2 = ttk.Frame(frame_opt)
        opts_frame2.pack(fill="x", pady=5)
        ttk.Label(opts_frame2, text="文件后缀:").pack(side="left", padx=5)
        ttk.Entry(opts_frame2, textvariable=self.suffix_var, width=15).pack(side="left", padx=5)
        ttk.Label(opts_frame2, text=f"运行设备: {DEVICE}").pack(side="right", padx=10)

        # 4. 执行按钮与状态
        self.btn_run = ttk.Button(self.root, text="🚀 开启综合后处理分析", command=self.start_task_thread)
        self.btn_run.pack(pady=10)
        
        self.log_area = tk.Text(self.root, height=12, width=85, state="disabled", background="#f8f8f8")
        self.log_area.pack(padx=20, pady=5)

    def log(self, message):
        self.log_area.config(state="normal")
        self.log_area.insert("end", message + "\n")
        self.log_area.see("end")
        self.log_area.config(state="disabled")
        self.root.update()

    def browse_folder(self):
        # 使用绝对路径，明确指定 parent
        current_path = os.path.abspath(os.getcwd())
        init_dir = os.path.join(current_path, "ivd_results")
        if not os.path.exists(init_dir): init_dir = current_path
        
        try:
            # 关键修复：显式传递 parent=self.root
            directory = filedialog.askdirectory(
                parent=self.root,
                initialdir=init_dir,
                title="选择训练结果文件夹"
            )
            if directory:
                directory = os.path.abspath(directory)
                self.selected_dir.set(directory)
                self.scan_models(directory)
        except Exception as e:
            messagebox.showerror("系统错误", f"无法打开对话框: {e}")

    def scan_models(self, directory):
        try:
            self.model_files = [f for f in os.listdir(directory) if f.endswith(".pth")]
            if not self.model_files:
                self.log(f"⚠️ 警告：在该目录下未找到 .pth 模型文件")
                self.model_combo["values"] = []
                self.selected_model.set("")
                return
            
            self.model_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
            self.model_combo["values"] = self.model_files
            self.model_combo.current(0)
            self.log(f"✅ 成功扫描到 {len(self.model_files)} 个模型文件")
        except Exception as e:
            self.log(f"❌ 扫描失败: {e}")

    def start_task_thread(self):
        if not self.selected_dir.get() or not self.selected_model.get():
            messagebox.showwarning("提示", "请选择结果文件夹和模型文件！")
            return
        
        self.btn_run.config(state="disabled")
        # 清除旧日志
        self.log_area.config(state="normal")
        self.log_area.delete("1.0", "end")
        self.log_area.config(state="disabled")
        
        thread = threading.Thread(target=self.run_visualization, daemon=True)
        thread.start()

    def run_visualization(self):
        selected_run_dir = self.selected_dir.get()
        model_name = self.selected_model.get()
        model_path = os.path.join(selected_run_dir, model_name)
        suffix = self.suffix_var.get()
        
        try:
            self.log(f"🔔 任务已启动...")
            self.log(f"📂 目录: {os.path.basename(selected_run_dir)}")
            self.log(f"🧠 模型: {model_name}")
            
            # 1. 实例化 Solver
            self.log("⚙️ 正在初始化求解器...")
            solver = Solver()
            
            # 2. 加载权重
            self.log("💾 正在载入模型权重...")
            state_dict = torch.load(model_path, map_location=DEVICE)
            
            # 智能兼容加载
            try:
                solver.model.load_state_dict(state_dict)
            except:
                self.log("⚠️ 权重结构不完全一致，尝试松散加载模式...")
                solver.model.load_state_dict(state_dict, strict=False)
            
            # 3. 创建专属后处理子文件夹 (对标 run_post.py)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            post_dir = os.path.join(selected_run_dir, f"Post_Analysis_{timestamp}")
            os.makedirs(post_dir, exist_ok=True)
            self.log(f"📁 创建文件夹: {f'Post_Analysis_{timestamp}'}")

            # 4. 后处理
            self.log("🎨 正在生成所选后处理内容...")
            self.log("   (包含高阶导数推断，请耐心等待 10-30 秒)")
            
            # 将结果输出到子文件夹
            if self.var_deformed.get() or self.var_undeformed.get() or self.var_vtk.get():
                generate_all_plots(solver, post_dir, suffix=suffix, 
                                   gen_deformed=self.var_deformed.get(),
                                   gen_undeformed=self.var_undeformed.get(),
                                   gen_vtk=self.var_vtk.get())
                                   
            if self.var_csv.get():
                self.log("📊 正在导出全量 CSV 结果数据...")
                from export_results import export_results
                csv_dir = os.path.join(post_dir, "CSV_Data")
                os.makedirs(csv_dir, exist_ok=True)
                export_results(model_path, save_dir=csv_dir)
            
            self.log(f"\n✨ 恭喜！综合后处理全部完成。")
            self.log(f"📍 结果已存入子目录: {os.path.basename(post_dir)}")
            messagebox.showinfo("成功", f"可视化结果已生成完毕！\n保存于: {os.path.basename(post_dir)}")
            
        except Exception as e:
            self.log(f"❌ 报错了: {str(e)}")
            messagebox.showerror("后处理失败", f"详细错误: {e}")
        
        finally:
            self.btn_run.config(state="normal")

if __name__ == "__main__":
    # 强制设置工作目录为脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    root = tk.Tk()
    app = VisualizerGUI(root)
    root.mainloop()
