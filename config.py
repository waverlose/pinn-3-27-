"""
配置
"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

GEO_A, GEO_B, GEO_H = 20.0, 15.0, 10.0
NP_AXIS_RATIO = 0.7

PHY_R, PHY_T = 8314.0, 310.0
PHY_C_EXT = 1.5e-7
VISCO_SI = 1e-3 
ION_RP, ION_RN = 0.2e-6, 0.14e-6
ION_DP0, ION_DN0 = 1.33e-3, 1.84e-3

ENABLE_ANISOTROPY = True
MAT_NP_LAMBDA, MAT_NP_MU, MAT_NP_WSR = 0.04, 0.06, 0.15 # 对齐 COMSOL: lambda=0.04, mu=0.06
MAT_AF_LAMBDA, MAT_AF_MU, MAT_AF_WSR = 0.17, 0.14, 0.3 # 纤维环参数保持对齐
MAT_FIBER_ALPHA_MAX = 20.0 # AF外边界最大纤维刚度系数，调整为20MPa
MAT_FIBER_THETA_INNER, MAT_FIBER_THETA_OUTER = 30.0, 30.0

MAT_NP_PERM_A, MAT_NP_PERM_N = 2.48e-5, 2.15
MAT_AF_PERM_A, MAT_AF_PERM_N = 3.10e-5, 2.20
MAT_NP_DIFF_A, MAT_NP_DIFF_B = 1.25, 0.68
MAT_AF_DIFF_A, MAT_AF_DIFF_B = 1.29, 0.37

BC_TOP_PRESSURE, BC_TOP_COMPRESSION = 0.3, 0.1  # 上边界压力
BC_TOP_FIX_HORIZONTAL, BC_SIDE_ROBIN_BETA = True, 0.5
LOSS_WEIGHT_SIDE_CHEM_SCALE = 100.0

OUTPUT_BASE_DIR = "ivd_results"
NET_HIDDEN_DIM, NET_NUM_LAYERS = 128, 6
TRAIN_TOTAL_EPOCHS = 15000 
DEBUG_STATS_FREQ = 200
TRAIN_BATCH_DOM = 5000   
TRAIN_BATCH_BC = 2000    
TRAIN_INIT_LR, TRAIN_LR_GAMMA = 2e-4, 0.99996 

RAS_ENABLED, RAS_FREQ = True, 100
RAS_NUM_CANDIDATES, RAS_MIX_RATIO = 10000, 0.1

# 特征尺度 (关键修正: 统一为 1.0，由自适应权重接管，消除梯度放大炸弹)
S_MOM = 1.0       
S_FLOW_W = 1.0  
S_FLOW_I = 1.0 

PRINT_FREQ, SAVE_FREQ = 100, 500
VIS_GRID_NX, VIS_GRID_NZ, VIS_DPI = 100, 50, 300
MODEL_OUT_SCALE_U, MODEL_OUT_SCALE_MU = 1.0, 1.0  # 位移缩放

# 权重系统配置 (Weight System Configuration)
WEIGHT_STRATEGY = "gradnorm"  # gradnorm, simple, fixed
WEIGHT_INIT = {
    "pde_mom": 1.0,
    "pde_flow": 0.01,  # 与 train.py 保持一致，修复初始化不一致问题
    "pde_ion": 0.01,
    "bc": 1.0  # 边界条件权重
}
WEIGHT_UPDATE_FREQ = 100  # 权重更新频率（epoch）
WEIGHT_SAMPLE_SIZE = 1000  # 梯度估计样本数
WEIGHT_CLAMP_RANGES = {
    "pde_mom": (0.1, 10.0),
    "pde_flow": (0.01, 20.0),
    "pde_ion": (0.01, 20.0),
    "bc": (1.0, 200.0)  # 边界条件权重范围
}
WEIGHT_MOMENTUM = 0.9  # 权重更新动量系数
WEIGHT_LBFGS_ENABLED = True  # L-BFGS阶段是否启用自适应权重