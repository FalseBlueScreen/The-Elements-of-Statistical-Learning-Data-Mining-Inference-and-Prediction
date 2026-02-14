# 第3章 回归的线性方法 (Linear Methods for Regression)

## 3.1 引言 (Introduction)
线性模型简单且通常提供了对数据的充分描述。对于小样本量（相对于输入维度 $p$）的问题，线性模型往往优于非线性模型。本章描述了线性回归模型及其扩展。

## 3.2 线性回归模型与最小二乘 (Linear Regression Models and Least Squares)
线性回归模型假设回归函数 $E(Y|X)$ 是输入的线性函数：
$$ f(X) = \beta_0 + \sum_{j=1}^p X_j \beta_j $$
通过最小化残差平方和 (RSS) 估计参数 $\beta$：
$$ RSS(\beta) = \sum_{i=1}^N (y_i - \beta_0 - \sum_{j=1}^p x_{ij} \beta_j)^2 $$
这可以写成矩阵形式：
$$ RSS(\beta) = (y - X\beta)^T (y - X\beta) $$
对 $\beta$ 求导并令其为零，得到正规方程 (Normal Equations)：
$$ X^T(y - X\beta) = 0 $$
如果 $X^T X$ 非奇异，解为：
$$ \hat{\beta} = (X^T X)^{-1} X^T y $$

### 几何解释 (Geometric Interpretation)
图3.7 展示了最小二乘估计的几何意义。我们将 $y$ 投影到由 $X$ 的列向量张成的子空间上，使得残差向量 $y - \hat{y}$ 正交于该子空间。

## 3.3 子集选择 (Subset Selection)
当预测变量很多时，为了提高预测精度和解释性，我们通常只选择一部分变量。
- **最优子集选择 (Best-Subset Selection)**: 遍历所有可能的 $2^p$ 个子集，选择RSS最小的模型。计算量极大。
- **逐步选择 (Stepwise Selection)**:
  - **向前逐步选择 (Forward Stepwise)**: 从空模型开始，每次添加最显著的变量。
  - **向后逐步选择 (Backward Stepwise)**: 从全模型开始，每次移除最不显著的变量。
  - **混合逐步选择 (Hybrid Stepwise)**: 结合两者。

## 3.4 收缩方法 (Shrinkage Methods)
子集选择是一个离散过程，往往导致较高的方差。收缩方法通过对系数大小施加惩罚，使系数连续地向零收缩。

### 3.4.1 岭回归 (Ridge Regression)
岭回归在最小二乘的基础上加上 $L2$ 惩罚项：
$$ \hat{\beta}^{ridge} = \text{argmin}_{\beta} \{ \sum_{i=1}^N (y_i - \beta_0 - \sum_{j=1}^p x_{ij} \beta_j)^2 + \lambda \sum_{j=1}^p \beta_j^2 \} $$
其中 $\lambda \ge 0$ 是调节参数。
岭回归的解为：
$$ \hat{\beta}^{ridge} = (X^T X + \lambda I)^{-1} X^T y $$
当 $X$ 列向量之间存在多重共线性时，岭回归非常有用。

### 3.4.2 Lasso (Least Absolute Shrinkage and Selection Operator)
Lasso 使用 $L1$ 惩罚项：
$$ \hat{\beta}^{lasso} = \text{argmin}_{\beta} \{ \frac{1}{2} \sum_{i=1}^N (y_i - \beta_0 - \sum_{j=1}^p x_{ij} \beta_j)^2 + \lambda \sum_{j=1}^p |\beta_j| \} $$
与岭回归不同，Lasso 的解是稀疏的（即许多系数正好为零），因此它同时实现了变量选择和收缩。

### 3.4.3 最小角回归 (Least Angle Regression, LARS)
LARS 是一种高效的算法，用于求解 Lasso 路径。它类似于向前逐步回归，但在每一步只增加相关变量的系数，直到另一个变量与当前残差的相关性相同。

## 3.5 衍生输入方向的方法 (Methods Using Derived Input Directions)
这些方法将原始输入变量组合成少量的新变量（通常是线性组合），然后在这些新变量上进行回归。
- **主成分回归 (Principal Components Regression, PCR)**: 使用输入数据的主成分作为新的预测变量。它主要通过方差最大化来选择方向，但这并不一定能最好地预测 $Y$。
- **偏最小二乘 (Partial Least Squares, PLS)**: 同时使用 $X$ 和 $Y$ 来寻找新的方向，使得这些方向既能很好地解释 $X$ 的变化，又能很好地预测 $Y$。

## 3.6 比较与讨论 (Discussion: A Comparison of the Selection and Shrinkage Methods)
岭回归和 Lasso 都可以看作是贝叶斯估计，分别对应高斯先验和拉普拉斯先验。
- 当真实的系数是稠密的（即许多小的非零系数）时，岭回归表现较好。
- 当真实的系数是稀疏的（即只有少数大的非零系数）时，Lasso 表现较好。
- 弹性网络 (Elastic Net) 结合了 $L1$ 和 $L2$ 惩罚，试图兼顾两者的优点。

---
*注：本章深入探讨了线性回归的变体，重点在于解决过拟合和变量选择问题。图3.5展示了Lasso系数随 $\lambda$ 变化的路径。*
