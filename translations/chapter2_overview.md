# 第2章 有监督学习概论 (Overview of Supervised Learning)

## 2.1 引言 (Introduction)
有监督学习的核心目标是基于一组输入变量（Input Variables）来预测输出变量（Output Variable）。
- **输入变量**：通常记为 $X$，也被称为预测变量 (Predictors)、特征 (Features)、自变量 (Independent Variables)。
- **输出变量**：通常记为 $Y$，也被称为响应变量 (Response)、因变量 (Dependent Variable)。

学习任务就是学习从输入到输出的映射函数 $f: X \to Y$。

### 变量类型与编码 (Variable Types and Terminology)
1. **定量变量 (Quantitative)**: 数值型，具有大小和顺序，如血压、股价。
2. **定性变量 (Qualitative)**: 类别型，如性别、血型。通常用 $K$ 个水平（Levels）或类别表示。
   - **编码**: 定性变量常通过指示变量（Dummy Variables）进行编码。例如，如果有 $K$ 个类别，可以定义 $K$ 个指示变量 $X_k \in \{0, 1\}$，其中 $X_k=1$ 表示属于第 $k$ 类。为了避免线性相关，通常只需要 $K-1$ 个变量。
3. **有序分类变量 (Ordered Categorical)**: 具有特定顺序的类别变量，如病情程度（轻、中、重）。

我们用 $X \in \mathbb{R}^p$ 表示 $p$ 维输入向量，$Y$ 表示输出。
- **回归 (Regression)**: 当 $Y$ 是定量变量时。
- **分类 (Classification)**: 当 $Y$ 是定性变量时。

## 2.2 两种简单的预测方法 (Two Simple Approaches to Prediction)

### 2.2.1 线性模型与最小二乘法 (Linear Models and Least Squares)
线性模型假设回归函数 $E(Y|X)$ 是线性的：
$$ \hat{Y} = \hat{\beta}_0 + \sum_{j=1}^p X_j \hat{\beta}_j = X^T \hat{\beta} $$
其中 $\hat{\beta}$ 是通过最小化残差平方和 (RSS) 得到的参数估计：
$$ RSS(\beta) = \sum_{i=1}^N (y_i - x_i^T \beta)^2 = (y - X\beta)^T (y - X\beta) $$
通过对 $\beta$ 求导并令为0，得到正规方程：
$$ X^T(y - X\beta) = 0 \implies \hat{\beta} = (X^T X)^{-1} X^T y $$
这意味着 $\hat{\beta}$ 是唯一的，只要 $X^T X$ 是非奇异的（满秩）。

### 2.2.2 线性回归 vs K-最近邻 (Linear Regression vs. Nearest-Neighbor Methods)
最近邻方法不假设 $f(X)$ 的具体形式，而是根据局部的数据点进行预测。
$$ \hat{Y}(x) = \frac{1}{k} \sum_{x_i \in N_k(x)} y_i $$
其中 $N_k(x)$ 是包含 $x$ 的 $k$ 个最近邻的集合。
- **线性模型**: 假设强（线性关系），偏差大，方差小。适用于样本量少、维度高、关系简单的情况。
- **最近邻**: 假设弱（局部平滑），偏差小，方差大（尤其是 $k$ 小时）。随着 $k \to N$，模型变为全局平均，偏差增加，方差减小。
- **有效参数个数**: 对于线性回归，参数个数为 $p+1$。对于 k-NN，有效参数个数约为 $N/k$。

## 2.3 统计决策理论 (Statistical Decision Theory)
我们需要一个定量的标准来评估预测函数 $f(X)$ 的好坏。为此引入**损失函数 (Loss Function)** $L(Y, f(X))$。

### 平方误差损失 (Squared Error Loss)
$$ EPE(f) = E[L(Y, f(X))] = E[(Y - f(X))^2] $$
通过条件期望分解：
$$ EPE(f) = E_X E_{Y|X} ([Y - f(X)]^2 | X) $$
为了最小化 EPE，只需对每个 $X=x$ 最小化内部的期望：
$$ f(x) = \text{argmin}_c E_{Y|X} ([Y - c]^2 | X=x) $$
解为条件期望：
$$ f(x) = E(Y | X=x) $$
这就是**回归函数**。

### 0-1 损失 (0-1 Loss)
对于分类问题，损失函数通常是：
$$ L(G, \hat{G}(X)) = I(G \neq \hat{G}(X)) $$
期望预测误差为错误率：
$$ EPE = E[I(G \neq \hat{G}(X))] $$
最小化该误差的解是**贝叶斯分类器 (Bayes Classifier)**：
$$ \hat{G}(x) = \text{argmax}_{g \in \mathcal{G}} P(G=g | X=x) $$
即选择后验概率最大的类别。

## 2.4 高维空间中的局部方法 (Local Methods in High Dimensions)
这一节讨论了**维数灾难 (Curse of Dimensionality)**。
在高维空间 ($p$ 很大) 中，数据变得极其稀疏。
1.  **邻域不再“局部”**: 为了捕获固定比例的数据（如10%），在高维空间中需要覆盖输入空间的很大一部分。例如，在单位超立方体中，捕获比例 $r$ 的样本所需的边长比例为 $e_p(r) = r^{1/p}$。当 $p=10, r=0.01$ 时，$e_{10}(0.01) = 0.63$。这意味着为了找到1%的邻居，我们需要覆盖各维度63%的范围，这不再是局部的。
2.  **数据倾向于分布在边界**: 随着维度增加，大部分数据点都位于样本空间的边界附近，而不是中心。这意味着预测点通常也是在训练数据的凸包边缘甚至外部（外推），而不是内部（内插）。
3.  **距离度量失效**: 在高维空间中，最近邻和最远邻的距离差异变小，所有点之间的距离趋于相等，使得基于距离的方法（如 k-NN）失效。

## 2.5 统计模型，有监督学习与函数逼近 (Statistical Models, Supervised Learning and Function Approximation)
统计学习可以看作是在某个函数空间 $\mathcal{H}$ 中寻找最佳的 $f(x)$ 来逼近真实关系。
由于数据有限，我们不能在任意复杂的空间中搜索，必须限制 $\mathcal{H}$ 的复杂度。
- **线性基扩展 (Linear Basis Expansion)**:
  $$ f_\theta(x) = \sum_{k=1}^K \theta_k h_k(x) $$
  其中 $h_k(x)$ 是基函数（如多项式、三角函数）。
- **最大似然估计 (Maximum Likelihood Estimation)**: 最小二乘法等价于假设误差服从高斯分布时的最大似然估计。
  $$ L(\theta) = \sum_{i=1}^N \log P_\theta(y_i | x_i) $$

## 2.6 结构化回归模型 (Structured Regression Models)
为了克服维数灾难，我们需要对回归函数 $f(X)$ 施加结构假设。
- **投影寻踪 (Projection Pursuit)**:
  $$ f(X) = \sum_{m=1}^M g_m(\alpha_m^T X) $$
  通过寻找特定的投影方向 $\alpha_m$ 来降低维度。
- **神经网络 (Neural Networks)**:
  可以看作是一种多层的非线性投影寻踪模型。
  $$ f(X) = \sigma(\alpha^T X) $$

## 2.7 限制性估计量的分类 (Classes of Restricted Estimators)
非参数回归技术主要分为三类：
1.  **粗糙度惩罚 (Roughness Penalty)**:
    $$ RSS(f) + \lambda J(f) $$
    其中 $J(f)$ 惩罚函数的复杂度（如二阶导数）。
2.  **核方法 (Kernel Methods)**:
    $$ \hat{f}(x) = \frac{\sum_{i=1}^N K_\lambda(x, x_i) y_i}{\sum_{i=1}^N K_\lambda(x, x_i)} $$
    通过核函数 $K_\lambda$ 显式地指定局部权重。
3.  **基函数方法 (Basis Functions)**:
    假设 $f$ 可以由一组基函数线性表示，如样条函数 (Splines)。

---
*本章总结了有监督学习的基本框架：从数据中学习输入到输出的映射。关键在于权衡模型的灵活性（偏差）与稳定性（方差），尤其是在高维空间中，必须引入结构假设来规避维数灾难。*
