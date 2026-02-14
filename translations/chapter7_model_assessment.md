# 第7章 模型评估与选择 (Model Assessment and Selection)

## 7.1 引言 (Introduction)
模型的泛化能力（Generalization Performance），即在独立测试数据上的预测能力，是评估学习方法的关键。本章介绍了评估模型预测性能的方法，并指导如何选择模型复杂度。

## 7.2 偏差、方差与模型复杂度 (Bias, Variance, and Model Complexity)
如果我们有足够多的数据，我们可以将它们分为三部分：
1. **训练集 (Training Set)**: 用于拟合模型。
2. **验证集 (Validation Set)**: 用于模型选择（估计预测误差）。
3. **测试集 (Test Set)**: 用于评估最终所选模型的泛化误差。

**测试误差 (Test Error)** 是对独立测试样本 $(X^0, Y^0)$ 的平均预测误差：
$$ Err_{\mathcal{T}} = E[L(Y, \hat{f}(X)) | \mathcal{T}] $$
其中 $\mathcal{T}$ 是固定的训练集。
**期望预测误差 (Expected Prediction Error, EPE)**（或泛化误差）是对所有可能的训练集取期望：
$$ Err = E[Err_{\mathcal{T}}] = E[L(Y, \hat{f}(X))] $$

**图7.1** 展示了随着模型复杂度的增加，训练误差持续下降，但测试误差呈现 U 形曲线（先降后升）。

## 7.3 偏差-方差分解 (The Bias-Variance Decomposition)
假设 $Y = f(X) + \epsilon$，其中 $E(\epsilon) = 0, Var(\epsilon) = \sigma_\epsilon^2$。
对于平方误差损失，期望预测误差可以分解为：
$$ Err(x_0) = \sigma_\epsilon^2 + Bias^2(\hat{f}(x_0)) + Var(\hat{f}(x_0)) $$
- **不可约误差 (Irreducible Error)** $\sigma_\epsilon^2$: 即使知道了真实的 $f(x)$，我们也无法消除这部分误差。
- **偏差 (Bias)**: 模型预测值的期望与真实值的差异。
  $$ Bias(\hat{f}(x_0)) = E[\hat{f}(x_0)] - f(x_0) $$
  通常，简单的模型（如线性模型）具有高偏差。
- **方差 (Variance)**: 模型预测值围绕其期望的波动程度。
  $$ Var(\hat{f}(x_0)) = E[(\hat{f}(x_0) - E[\hat{f}(x_0)])^2] $$
  复杂的模型（如 k=1 KNN）具有高方差。

通常，增加模型复杂度会减少偏差但增加方差。好的模型需要在两者之间取得平衡。
- 对于线性模型，偏差与模型的曲率（非线性程度）有关，方差与参数个数 $p$ 有关：$Var \approx p \sigma_\epsilon^2 / N$。
- 对于 k-NN，偏差随 $k$ 增大而增大，方差随 $k$ 增大而减小（$Var \approx \sigma_\epsilon^2 / k$）。

## 7.4 训练误差的乐观性 (The Optimism of the Training Error Rate)
训练误差通常小于真实的测试误差，因为同一数据被用于拟合和评估。这种现象称为**乐观性 (Optimism)**。
$$ \text{op} = Err_{in} - \overline{err} $$
其中 $Err_{in}$ 是样本内误差（In-sample Error），$\overline{err}$ 是训练误差。
对于线性模型和其他简单的加性模型，乐观性与参数个数 $d$ 成正比：
$$ E[\text{op}] \approx \frac{2}{N} \sum_{i=1}^N Cov(\hat{y}_i, y_i) $$
如果 $\hat{y}_i$ 是由具有 $d$ 个参数的线性拟合得到的，那么 $\sum Cov(\hat{y}_i, y_i) = d \sigma_\epsilon^2$。
所以：
$$ E[\text{op}] \approx \frac{2d}{N} \sigma_\epsilon^2 $$
这意味着由于模型使用了数据来估计自身参数，它对训练数据的拟合会“过于好”。这就引出了我们需要惩罚模型复杂度的原因。

## 7.5 样本内预测误差的估计 (Estimates of In-Sample Prediction Error)
为了选择模型，我们可以估计样本内误差 $Err_{in}$。常用的方法有：
- **$C_p$ 统计量 (Mallows' $C_p$)**:
  $$ C_p = \overline{err} + 2 \frac{d}{N} \hat{\sigma}_\epsilon^2 $$
  这里 $d$ 是有效参数个数。
- **AIC (Akaike Information Criterion)**:
  $$ AIC = -2 \log L(\hat{\theta}) + 2d $$
  对于高斯模型，$AIC$ 与 $C_p$ 等价。AIC 来源于信息论中的 KL 散度。
- **BIC (Bayesian Information Criterion)**:
  $$ BIC = -2 \log L(\hat{\theta}) + d \log N $$
  BIC 对复杂模型的惩罚比 AIC 重（因为 $\log N > 2$ for $N > 7$），因此 BIC 倾向于选择更简单的模型。如果真实模型在候选集合中，BIC 选择正确模型的概率随 $N \to \infty$ 趋向于 1（一致性）。

## 7.6 交叉验证 (Cross-Validation)
交叉验证是一种直接估计泛化误差 $Err$ 的方法，不依赖于关于误差分布的假设。
**K-折交叉验证 (K-Fold Cross-Validation)**:
1. 将数据随机分成 $K$ 个大小相等的子集（折）。
2. 对于 $k=1, \dots, K$：
   - 使用除第 $k$ 折以外的所有数据训练模型。
   - 在第 $k$ 折上计算预测误差。
3. 计算 $K$ 次误差的平均值。
$$ CV(\hat{f}) = \frac{1}{N} \sum_{i=1}^N L(y_i, \hat{f}^{-k(i)}(x_i)) $$
常用的 $K$ 值是 5 或 10。
- 当 $K=N$ 时（留一法 Leave-One-Out, LOOCV），估计几乎无偏，但方差很大且计算量大。
- 当 $K=5$ 或 10 时，方差较小，但可能有轻微的偏差（高估误差，因为训练集比全集小）。

**广义交叉验证 (Generalized Cross-Validation, GCV)**:
它是 LOOCV 的一种加权版本，对于线性平滑矩阵 $S$：
$$ GCV(\hat{f}) = \frac{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{f}(x_i))^2}{(1 - \text{trace}(S)/N)^2} $$
GCV 具有旋转不变性，且计算通常比 LOOCV 快。

## 7.7 自助法 (Bootstrap Methods)
自助法通过从原始数据中有放回地抽样生成许多个“自助样本”，在这些样本上训练模型，并在原始样本上评估。
标准的自助法估计通常向上偏倚（悲观），因为自助样本只包含原始数据中约 $1 - (1 - 1/N)^N \approx 1 - e^{-1} \approx 63.2\%$ 的独特观测值。这意味着自助样本的“有效样本量”只有原始数据的 63.2%。

**.632 估计量**:
$$ \hat{Err}^{(.632)} = 0.368 \cdot \overline{err} + 0.632 \cdot \hat{Err}^{(1)} $$
这试图纠正这种偏差，通过加权平均训练误差（通常向下偏）和留一法自助误差（通常向上偏）。
然而，如果过拟合非常严重（如 1-NN），$\overline{err}=0$，.632 估计量仍然可能低估误差。
**.632+ 估计量**:
为了进一步修正过拟合问题，提出了 .632+ 估计量，它根据相对过拟合率调整权重。

---
*注：本章讨论了模型选择的核心问题。图7.1展示了模型复杂度与误差的关系，是机器学习中最重要的概念图之一。我们介绍了 AIC/BIC 用于模型选择，以及 CV 和 Bootstrap 用于误差估计。*
