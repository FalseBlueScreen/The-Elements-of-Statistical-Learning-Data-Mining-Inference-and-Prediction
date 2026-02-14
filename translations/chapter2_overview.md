# 第2章 有监督学习概论 (Overview of Supervised Learning)

## 2.1 引言 (Introduction)

有监督学习的目标是利用一组输入变量（input variables）来预测输出变量（output variable）。输入变量通常记为 $X$，也被称为预测变量（predictors）、特征（features）或自变量。输出变量通常记为 $Y$，也被称为响应变量（response）或因变量。

### 变量类型与术语 (Variable Types and Terminology)
- **定量变量 (Quantitative)**: 取值为数值，具有大小和顺序，如血压、股价。
- **定性变量 (Qualitative)**: 取值为类别，无大小顺序，如性别（男/女）、血型（A/B/O/AB）。定性变量通常用数字编码表示，但这些数字没有数学意义。
- **有序分类变量 (Ordered Categorical)**: 介于两者之间，如病情严重程度（轻/中/重）。

我们用 $X \in \mathbb{R}^p$ 表示输入向量，$Y$ 表示输出。如果是回归问题，$Y$ 是定量的；如果是分类问题，$Y$ 是定性的。

## 2.2 两种简单的预测方法 (Two Simple Approaches to Prediction)

### 2.2.1 线性模型与最小二乘法 (Linear Models and Least Squares)
线性模型假设回归函数 $E(Y|X)$ 是输入的线性函数：
$$ \hat{Y} = \hat{\beta}_0 + \sum_{j=1}^p X_j \hat{\beta}_j $$
其中 $\hat{\beta}$ 是参数。我们通常通过最小化残差平方和（Residual Sum of Squares, RSS）来估计参数：
$$ RSS(\beta) = \sum_{i=1}^N (y_i - x_i^T \beta)^2 $$
线性模型简单且易于解释，但在很多情况下，真实的函数关系并非线性的。

### 2.2.2 K-最近邻法 (Nearest-Neighbor Methods)
最近邻法不依赖于严格的模型结构假设。对于给定的查询点 $x$，它查找训练集中距离 $x$ 最近的 $k$ 个观测点，并用这些点的平均值作为预测值：
$$ \hat{Y}(x) = \frac{1}{k} \sum_{x_i \in N_k(x)} y_i $$
其中 $N_k(x)$ 是包含 $k$ 个最近邻的集合。
- 当 $k=1$ 时，模型非常灵活，拟合误差为0，但容易过拟合（高方差）。
- 当 $k$ 很大时，模型变得平滑，但偏差增加。

**图2.1** 展示了这两种方法的对比。线性决策边界平滑且稳定，而1-最近邻的决策边界非常不规则，紧紧包围着训练数据。

## 2.3 统计决策理论 (Statistical Decision Theory)
我们需要一个框架来评估预测的好坏。我们引入**损失函数 (Loss Function)** $L(Y, \hat{f}(X))$。
最常用的损失函数是平方误差损失 (Squared Error Loss): $L(Y, f(X)) = (Y - f(X))^2$。
在该损失函数下，最优预测函数是条件期望：
$$ f(x) = E(Y | X=x) $$
这被称为**回归函数 (Regression Function)**。

对于分类问题，常用的损失函数是0-1损失。此时，最优预测器是贝叶斯分类器 (Bayes Classifier)：
$$ \hat{G}(x) = \text{argmax}_{g \in \mathcal{G}} P(G=g | X=x) $$

## 2.4 高维空间中的局部方法 (Local Methods in High Dimensions)
这一节讨论了著名的**维数灾难 (Curse of Dimensionality)**。
在高维空间中，数据变得非常稀疏。为了捕获局部结构（如最近邻方法所需的），我们必须覆盖输入空间的很大一部分体积，但这会导致邻域不再具有“局部”性。
例如，在单位立方体中，如果要捕获10%的数据，在1维空间中只需覆盖10%的长度；而在10维空间中，需要覆盖 $0.1^{1/10} \approx 0.8$ 即80%的边长。这意味着我们的“局部”邻域实际上延伸到了整个空间的边缘。

这种稀疏性导致了以下问题：
1. 样本密度随着维度增加呈指数级下降。
2. 即使样本量很大，高维空间中的数据点也倾向于位于边界附近。
3. 高维空间中的距离度量变得不再具有区分度。

## 2.5 统计模型，有监督学习与函数逼近
统计模型试图通过限制函数空间来解决维数灾难。例如，加性模型 (Additive Models) 假设：
$$ f(X) = \sum_{j=1}^p f_j(X_j) $$
这种假设大大降低了估计的难度，但牺牲了模型捕捉变量间复杂交互作用的能力。

## 2.6 结构化回归模型 (Structured Regression Models)
除了线性模型和最近邻，还有许多中间方法，如样条平滑 (Spline Smoothing)、核平滑 (Kernel Smoothing) 等，它们通过引入平滑度惩罚或限制基函数的数量来控制模型的复杂度。

## 2.7 模型选择与偏差-方差权衡 (Model Selection and the Bias-Variance Tradeoff)
所有模型都面临偏差（Bias）与方差（Variance）的权衡。
- **简单模型**（如线性回归）：偏差高，方差低。
- **复杂模型**（如k=1的KNN）：偏差低，方差高。
我们需要选择适当的模型复杂度，以最小化测试误差 (Test Error)。

---
*注：本章通过对比线性回归和K-最近邻，引入了统计学习的基本概念。图2.1至2.5形象地展示了这些方法的特点及高维问题。*
