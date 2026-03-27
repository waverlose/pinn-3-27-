# 弹性公式对比

## COMSOL 公式
```
Wmatrix = 0.5*mu*(I1C-3) - mu*log(J) + lambda*(1-wsr)^2*((J-1)/(1-wsr) - log((J-wsr)/(1-wsr)))
```
其中：
- I1C = C11+C22+C33 = B11+B22+B33 (因为 C = F^T F, B = F F^T)
- J = det(F)
- wsr = 固体体积分数（参考状态）
- phi0 = 1-wsr = 孔隙体积分数

## PINN 当前公式
neo-Hookean + 孔隙压力：
```
p_el = lam_s * phi0 * (J - 1.0) / (J - (1.0 - phi0))
tm = mu_s / J
sigma = tm*(B-I) + p_el*I
```

## 需要实现 COMSOL 公式

推导 PK2 应力：
S = 2*dW/dC

对于 COMSOL 公式：
```
dW/dI1C = 0.5*mu
dW/dJ = -mu/J + lambda*(1-wsr)^2*(1/(1-wsr) - 1/(J-wsr))
```

因为 dJ/dC = 0.5*J*C^{-1}
dI1C/dC = I

所以：
```
S = 2*(dW/dI1C * dI1C/dC + dW/dJ * dJ/dC)
  = mu*I + J*(-mu/J + lambda*(1-wsr)^2*(1/(1-wsr) - 1/(J-wsr)))*C^{-1}
  = mu*I + (-mu + lambda*(1-wsr)^2*(J/(1-wsr) - J/(J-wsr)))*C^{-1}
```

然后转换到 PK1: P = F*S

在 PINN 中已经用 B 而不是 C，需要调整。