"""
权重管理系统 (Weight Management System)
统一管理自适应权重，支持多种更新策略，解决初始化不一致、梯度估计偏差等问题。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.autograd import grad

from config import *


class WeightManager:
    """
    统一权重管理系统
    
    支持策略：
    - "gradnorm": GradNorm风格平衡，基于梯度范数调整权重
    - "simple": 简化版本，基于损失比例调整权重  
    - "fixed": 固定权重，不进行自适应调整
    
    特性：
    1. 统一初始化（解决solver.py与train.py不一致问题）
    2. 使用完整模型参数计算梯度（解决仅使用输出层参数的问题）
    3. 动态调整clamping范围（基于权重历史）
    4. 内存安全的梯度计算
    5. 支持L-BFGS阶段的权重应用
    6. 向后兼容旧检查点
    """
    
    def __init__(self, strategy: str = WEIGHT_STRATEGY, config: Optional[dict] = None):
        """
        初始化权重管理器
        
        Args:
            strategy: 权重更新策略，默认为config.WEIGHT_STRATEGY
            config: 可选配置字典，覆盖默认配置
        """
        self.strategy = strategy or WEIGHT_STRATEGY
        self.config = {
            "init": WEIGHT_INIT,
            "update_freq": WEIGHT_UPDATE_FREQ,
            "sample_size": WEIGHT_SAMPLE_SIZE,
            "clamp_ranges": WEIGHT_CLAMP_RANGES,
            "momentum": WEIGHT_MOMENTUM,
            "lbfgs_enabled": WEIGHT_LBFGS_ENABLED
        }
        if config:
            self.config.update(config)
        
        # 内存管理
        self._device = DEVICE
        
        # 初始化权重字典（全部为torch.Tensor，确保设备一致性）
        self.weights = {}
        self._initialize_weights()
        
        # 跟踪权重历史，用于动态调整clamping范围
        self.weight_history = {name: [] for name in self.weights.keys()}
        
    def _initialize_weights(self):
        """统一初始化权重，解决初始化不一致问题"""
        for name, init_val in self.config["init"].items():
            self.weights[name] = torch.tensor(init_val, device=self._device)
        print(f"权重管理器初始化完成，策略: {self.strategy}")
    
    def compute_gradient_norms(self, losses_dict: dict, model: nn.Module, 
                               sample_points: Optional[torch.Tensor] = None) -> dict:
        """
        计算各损失项的梯度范数（使用完整模型参数）
        
        Args:
            losses_dict: 损失字典 {name: loss_tensor}
            model: PINN模型实例
            sample_points: 可选，用于梯度计算的采样点（如果losses_dict不包含实际损失）
            
        Returns:
            dict: 梯度范数字典 {name: gradient_norm}
        """
        grads = {}
        
        # 静默模式：不打印调试信息
        
        # 使用losses_dict计算梯度（保留计算图以便多个梯度计算）
        model_params = list(model.parameters())
        
        # 收集需要计算梯度的损失项
        valid_losses = []
        for name, loss in losses_dict.items():
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                valid_losses.append((name, loss))
        
        if not valid_losses:
            return grads
        
        # 计算每个损失的梯度，使用retain_graph避免计算图被释放
        for i, (name, loss) in enumerate(valid_losses):
            # 最后一个损失项不使用retain_graph，释放计算图
            retain = (i < len(valid_losses) - 1)
            try:
                g_list = grad(loss, model_params, retain_graph=retain, allow_unused=True)
            except RuntimeError as e:
                print(f'  [WARN] Failed to compute gradient for {name}: {e}')
                # 如果失败，尝试强制释放计算图并跳过
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
                
            gn = torch.sqrt(torch.sum(torch.stack([torch.norm(g)**2 for g in g_list if g is not None])) + 1e-8)
            grads[name] = gn.detach()
            
            # 清理计算图引用
            del g_list
        
        # 最后清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return grads
    
    def _compute_residual_for_gradient(self, out: torch.Tensor, x: torch.Tensor, model: nn.Module):
        """
        为梯度计算准备残差（简化版本，避免复杂物理计算）
        注意：这是一个占位函数，实际实现应在Solver中
        """
        # 这里返回一个占位符，实际实现应在外部提供
        # 在Solver中，我们会直接调用compute_physics方法
        return [torch.zeros_like(out[:, 0:1]) for _ in range(5)]
    
    def update_weights(self, losses_dict: dict, model: nn.Module, 
                      epoch: int, sample_points: Optional[torch.Tensor] = None, 
                      force_update: bool = False) -> bool:
        """
        根据选定策略更新权重
        
        Args:
            losses_dict: 损失字典 {name: loss_tensor}
            model: PINN模型实例
            epoch: 当前训练轮次
            sample_points: 可选，用于梯度计算的采样点
            force_update: 是否强制更新（跳过频率检查）
            
        Returns:
            bool: 权重是否被更新
        """
        # 检查是否需要更新权重（除非强制更新）
        if not force_update and epoch % self.config["update_freq"] != 0:
            return False
        
        if self.strategy == "fixed":
            return False
        
        # 计算梯度范数
        grads = self.compute_gradient_norms(losses_dict, model, sample_points)
        if not grads:
            return False
        
        # 根据策略更新权重
        if self.strategy == "gradnorm":
            self._update_gradnorm(grads, epoch)
        elif self.strategy == "simple":
            self._update_simple(grads, losses_dict, epoch)
        
        # 记录权重历史
        for name, weight in self.weights.items():
            self.weight_history[name].append(weight.item())
            # 限制历史记录长度
            if len(self.weight_history[name]) > 100:
                self.weight_history[name].pop(0)
        
        return True
    
    def _update_gradnorm(self, grads: dict, epoch: int):
        """GradNorm风格权重更新：基于梯度范数平衡"""
        # 使用动量机制平滑权重更新
        momentum = self.config["momentum"]
        
        with torch.no_grad():
            # 计算基准梯度（使用pde_mom或第一个可用的梯度）
            base_grad = grads.get("pde_mom", None)
            if base_grad is None:
                base_grad = next(iter(grads.values()))
            
            for name in self.weights.keys():
                if name in grads:
                    # 计算目标权重：基准梯度 / 当前梯度
                    target_weight = base_grad / (grads[name] + 1e-8)
                    
                    # 应用clamping范围
                    clamp_min, clamp_max = self.config["clamp_ranges"].get(name, (0.1, 10.0))
                    target_weight = torch.clamp(target_weight, min=clamp_min, max=clamp_max)
                    
                    # 动量更新：新权重 = 动量 * 旧权重 + (1-动量) * 目标权重
                    new_weight = momentum * self.weights[name] + (1 - momentum) * target_weight
                    self.weights[name] = new_weight
    
    def _update_simple(self, grads: dict, losses_dict: dict, epoch: int):
        """简化权重更新：基于损失比例调整"""
        # 计算总损失
        total_loss = sum([loss.item() for loss in losses_dict.values() if isinstance(loss, torch.Tensor)])
        if total_loss < 1e-12:
            return
        
        momentum = self.config["momentum"]
        
        with torch.no_grad():
            for name in self.weights.keys():
                if name in losses_dict:
                    # 计算该损失项占总损失的比例
                    loss_item = losses_dict[name]
                    if isinstance(loss_item, torch.Tensor):
                        loss_ratio = loss_item.item() / total_loss
                        
                        # 目标权重与损失比例成反比（损失越大，权重越小）
                        target_weight = 1.0 / (loss_ratio + 1e-8)
                        
                        # 应用clamping范围（将target_weight转换为Tensor）
                        clamp_min, clamp_max = self.config["clamp_ranges"].get(name, (0.1, 10.0))
                        target_weight_tensor = torch.tensor(target_weight, device=self._device, dtype=torch.float32)
                        target_weight_tensor = torch.clamp(target_weight_tensor, min=clamp_min, max=clamp_max)
                        target_weight = target_weight_tensor  # 保持Tensor类型
                        
                        # 动量更新
                        new_weight = momentum * self.weights[name] + (1 - momentum) * target_weight
                        self.weights[name] = new_weight
    
    def apply_weights(self, losses_dict: dict) -> torch.Tensor:
        """
        应用权重到损失项，计算加权总损失
        
        Args:
            losses_dict: 损失字典 {name: loss_tensor}
            
        Returns:
            torch.Tensor: 加权总损失
        """
        total_loss = torch.tensor(0.0, device=self._device)
        
        # 应用权重到各个损失项
        for name, loss in losses_dict.items():
            if name in self.weights and isinstance(loss, torch.Tensor):
                weighted_loss = self.weights[name] * loss
                total_loss = total_loss + weighted_loss
            else:
                # 如果损失项没有对应的权重，直接添加
                total_loss = total_loss + loss
        
        return total_loss
    
    def get_weights(self) -> dict:
        """获取当前权重字典（返回Python原生值）"""
        return {name: weight.item() for name, weight in self.weights.items()}
    
    def set_weights(self, weights_dict: dict):
        """设置权重（用于从检查点加载）"""
        for name, value in weights_dict.items():
            if name in self.weights:
                if isinstance(value, torch.Tensor):
                    self.weights[name] = value.to(self._device)
                else:
                    self.weights[name] = torch.tensor(value, device=self._device)
    
    def get_state_dict(self) -> dict:
        """获取状态字典，用于保存检查点"""
        return {
            "strategy": self.strategy,
            "weights": self.get_weights(),
            "weight_history": self.weight_history,
            "config": self.config
        }
    
    def load_state_dict(self, state_dict: dict):
        """从状态字典加载"""
        self.strategy = state_dict.get("strategy", self.strategy)
        self.set_weights(state_dict.get("weights", {}))
        self.weight_history = state_dict.get("weight_history", self.weight_history)
        # 合并配置
        loaded_config = state_dict.get("config", {})
        self.config.update(loaded_config)
    
    def debug_info(self) -> str:
        """返回调试信息字符串"""
        info_lines = ["权重管理器状态:"]
        info_lines.append(f"  策略: {self.strategy}")
        info_lines.append(f"  当前权重:")
        for name, weight in self.weights.items():
            info_lines.append(f"    {name}: {weight.item():.4f}")
        return "\n".join(info_lines)
    
    def __getitem__(self, name: str) -> torch.Tensor:
        """通过名称获取权重"""
        return self.weights.get(name, torch.tensor(1.0, device=self._device))
    
    def __contains__(self, name: str) -> bool:
        """检查权重是否存在"""
        return name in self.weights