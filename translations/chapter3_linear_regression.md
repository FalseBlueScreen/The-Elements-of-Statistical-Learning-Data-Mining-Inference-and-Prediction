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

### 3.2.1 几何解释 (Geometric Interpretation)
图3.7 展示了最小二乘估计的几何意义。
- **空间 (Subspace)**: 我们将观测值 $y$ 投影到由 $X$ 的列向量张成的 $p+1$ 维子空间上。
- **正交性 (Orthogonality)**: 残差向量 $e = y - \hat{y}$ 正交于该子空间，即 $X^T e = 0$。
- **非奇异性 (Non-singularity)**: 当 $N > p$ 且 $X$ 列线性无关时，$X^T X$ 也是满秩的。如果存在多重共线性，$X^T X$ 接近奇异，$\beta$ 的方差会很大。

### 3.2.2 抽样性质 (Sampling Properties of $\beta$)
假设 $y_i = x_i^T \beta + \epsilon_i$，且 $\epsilon \sim N(0, \sigma^2 I)$。
- **期望**: $E(\hat{\beta}) = \beta$ (无偏估计)。
- **方差**: $Var(\hat{\beta}) = (X^T X)^{-1} \sigma^2$。
- **Z-分数 (Z-score)**: $z_j = \frac{\hat{\beta}_j}{\hat{\sigma} \sqrt{v_{jj}}}$，用于检验 $\beta_j=0$ 的假设。其中 $v_{jj}$ 是 $(X^T X)^{-1}$ 的对角元素。

### 3.2.3 高斯-马尔可夫定理 (Gauss-Markov Theorem)
在所有线性无偏估计量 $\tilde{\beta} = c^T y$ 中，最小二乘估计量 $\hat{\beta}$ 具有最小的方差。即对于任意向量 $a$，
$$ Var(a^T \hat{\beta}) \le Var(a^T \tilde{\beta}) $$
这就是著名的 **BLUE** (Best Linear Unbiased Estimator)。

### 3.2.4 多重回归源于简单一元回归 (Multiple Regression from Simple Univariate Regression)
可以通过 **格拉姆-施密特正交化 (Gram-Schmidt Orthogonalization)** 理解多元回归。
如果我们想计算 $\beta_p$ 的系数，我们可以先把 $x_p$ 和 $y$ 分别对之前的变量 $x_0, \dots, x_{p-1}$ 做回归，得到残差 $z_p$ 和 $r$。那么多元回归中 $\beta_p$ 的估计值实际上就是单变量回归 $\hat{\beta}_p = \frac{\langle z_p, r \rangle}{\langle z_p, z_p \rangle}$。
这也解释了为什么相关变量会导致估计不稳定：如果在之前的回归中 $x_p$ 已经被很好地解释了（残差 $z_p$ 很小），分母接近0，估计值就会很不稳定。

## 3.3 子集选择 (Subset Selection)
当预测变量很多时，为了提高预测精度和解释性，我们通常只选择一部分变量。
### 3.3.1 最优子集选择 (Best-Subset Selection)
遍历所有可能的 $2^p$ 个子集，选择RSS最小的模型。
- **计算复杂度**: $O(2^p)$，随着 $p$ 增加呈指数级增长。
- **跃迁**: 这种方法通过“跳跃和界限 (Leaps and Bounds)”算法可以加速，但仍然很难处理 $p > 40$ 的情况。

### 3.3.2 逐步选择 (Stepwise Selection)
- **向前逐步选择 (Forward Stepwise)**: 从空模型开始，每次添加最能降低 RSS 的变量。计算量为 $O(p^2)$。
- **向后逐步选择 (Backward Stepwise)**: 从全模型开始，每次移除最不显著的变量（即使 RSS 增加最少的变量）。只能用于 $N > p$。
- **混合逐步选择 (Hybrid Stepwise)**: 结合两者。

## 3.4 收缩方法 (Shrinkage Methods)
子集选择是一个离散过程，往往导致较高的方差。收缩方法通过对系数大小施加惩罚，使系数连续地向零收缩。

### 3.4.1 岭回归 (Ridge Regression)
岭回归在最小二乘的基础上加上 $L2$ 惩罚项：
$$ \hat{\beta}^{ridge} = \text{argmin}_{\beta} \{ \sum_{i=1}^N (y_i - \beta_0 - \sum_{j=1}^p x_{ij} \beta_j)^2 + \lambda \sum_{j=1}^p \beta_j^2 \} $$
解为：
$$ \hat{\beta}^{ridge} = (X^T X + \lambda I)^{-1} X^T y $$
**SVD 分解**:
如果 $X = U D V^T$，那么岭回归的拟合值为：
$$ X \hat{\beta}^{ridge} = \sum_{j=1}^p u_j \frac{d_j^2}{d_j^2 + \lambda} u_j^T y $$
这意味着岭回归对不同方向的主成分进行了不同程度的收缩（Shrinkage）。对于方差较小的主成分（$d_j$ 小），收缩更剧烈。
自由度 (Degrees of Freedom): $df(\lambda) = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}$。

### 3.4.2 Lasso (Least Absolute Shrinkage and Selection Operator)
Lasso 使用 $L1$ 惩罚项：
$$ \hat{\beta}^{lasso} = \text{argmin}_{\beta} \{ \frac{1}{2} \sum_{i=1}^N (y_i - \beta_0 - \sum_{j=1}^p x_{ij} \beta_j)^2 + \lambda \sum_{j=1}^p |\beta_j| \} $$
- **稀疏性**: Lasso 的解是稀疏的（即许多系数正好为零），因此它同时实现了变量选择和收缩。
- **几何解释**: $L1$ 范数对应的约束区域是菱形（多面体），其顶点位于坐标轴上。当最小二乘解的等高线首次接触约束区域时，往往接触在顶点处，此时对应的系数为0。

### 3.4.3 最小角回归 (Least Angle Regression, LARS)
LARS 是一种高效的算法，用于求解 Lasso 路径。
1. 从所有系数为0开始。
2. 找到与残差相关性最大的变量 $x_{j_1}$。
3. 沿着该变量方向移动，直到另一个变量 $x_{j_2}$ 与当前残差的相关性相同。
4. 此时，沿着 $x_{j_1}$ 和 $x_{j_2}$ 的角平分线方向移动，直到第三个变量加入。
LARS 的每一步计算量与最小二乘相当，只需 $p$ 步即可得到完整的 Lasso 路径。

## 3.5 衍生输入方向的方法 (Methods Using Derived Input Directions)
这些方法将原始输入变量组合成少量的新变量（通常是线性组合），然后在这些新变量上进行回归。
### 3.5.1 主成分回归 (Principal Components Regression, PCR)
使用输入数据 $X$ 的前 $M$ 个主成分 $Z_m = X v_m$ 作为新的预测变量。
- **无监督**: 主成分仅依赖于 $X$ 的协方差结构，不利用 $y$ 的信息。
- **降维**: 将 $p$ 维问题转化为 $M$ 维问题 ($M < p$)。
- **收缩**: 类似于岭回归，PCR 丢弃了方差最小的 $p-M$ 个成分（相当于将系数收缩为0），保留方差最大的 $M$ 个（不收缩）。

### 3.5.2 偏最小二乘 (Partial Least Squares, PLS)
同时使用 $X$ 和 $Y$ 来寻找新的方向。
$$ \hat{\phi}_m = X^T y $$
即寻找与 $y$ 协方差最大的方向。然后将 $X$ 投影到该方向上，并在残差上重复此过程。
- **有监督**: PLS 试图找到既能解释 $X$ 又能预测 $y$ 的方向。
- **方差-偏差权衡**: PLS 通常比 PCR 具有更低的偏差（因为它利用了 $y$），但方差可能更高。

## 3.6 比较与讨论 (Discussion: A Comparison of the Selection and Shrinkage Methods)
岭回归和 Lasso 都可以看作是贝叶斯估计，分别对应高斯先验和拉普拉斯先验。
- 当真实的系数是稠密的（即许多小的非零系数）时，岭回归表现较好。
- 当真实的系数是稀疏的（即只有少数大的非零系数）时，Lasso 表现较好。
- **弹性网络 (Elastic Net)**: 结合了 $L1$ 和 $L2$ 惩罚，试图兼顾两者的优点：即具有稀疏性（像 Lasso），又能处理相关变量群（像 Ridge，会将相关变量一起选入或剔除，而不是像 Lasso 那样随机选一个）。

---
*注：本章深入探讨了线性回归的变体，重点在于解决过拟合和变量选择问题。图3.5展示了Lasso系数随 $\lambda$ 变化的路径。*
