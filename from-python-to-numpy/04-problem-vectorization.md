# 问题向量化 (Problem vectorization)

## 简介 (Introduction)

问题向量化比代码向量化要困难得多，因为它意味着你基本上必须重新思考你的问题，以使其可向量化。大多数时候，这意味着你必须使用不同的算法来解决你的问题，甚至更糟……发明一个新的算法。因此，困难在于跳出框框思考。

为了说明这一点，让我们考虑一个简单的问题，给定两个向量 X 和 Y，我们要计算所有索引对 i, j 的 X[i]*Y[j] 的总和。一种简单而明显的解决方案是编写：

```python
def compute_python(X, Y):
    result = 0
    for i in range(len(X)):
        for j in range(len(Y)):
            result += X[i] * Y[j]
    return result
```

然而，这第一个朴素的实现需要两个循环，我们已经知道它会很慢：

```python
import numpy as np
from tools import timeit
X = np.arange(1000)
timeit("compute_python(X,X)", globals())
# 1 loops, best of 3: 0.274481 sec per loop
```

那么如何向量化这个问题呢？如果你还记得你的线性代数课程，你可能已经识别出表达式 X[i] * Y[j] 与矩阵乘积表达式非常相似。所以也许我们可以从一些 numpy 加速中受益。一种错误的解决方案是编写：

```python
def compute_numpy_wrong(X, Y):
    return (X*Y).sum()
```

这是错误的，因为 X*Y 表达式实际上会计算一个新的向量 Z，使得 Z[i] = X[i] * Y[i]，这不是我们想要的。相反，我们可以通过首先重塑两个向量然后将它们相乘来利用 numpy 广播：

```python
def compute_numpy(X, Y):
    Z = X.reshape(len(X),1) * Y.reshape(1,len(Y))
    return Z.sum()
```

这里我们有 Z[i,j] == X[i,0]*Y[0,j]，如果我们对 Z 的每个元素求和，我们就得到了预期的结果。让我们看看在这个过程中我们获得了多少加速：

```python
X = np.arange(1000)
timeit("compute_numpy(X,X)", globals())
# 10 loops, best of 3: 0.00157926 sec per loop
```

这更好，我们获得了约 150 倍的加速。但我们可以做得更好。

如果您再次仔细查看纯 Python 版本，您会发现内部循环使用的是不依赖于 j 索引的 X[i]，这意味着它可以从内部循环中移除。代码可以重写为：

```python
def compute_numpy_better_1(X, Y):
    result = 0
    for i in range(len(X)):
        Ysum = 0
        for j in range(len(Y)):
            Ysum += Y[j]
        result += X[i]*Ysum
    return result
```

但由于内部循环不依赖于 i 索引，我们也可以只计算一次：

```python
def compute_numpy_better_2(X, Y):
    result = 0
    Ysum = 0
    for j in range(len(Y)):
        Ysum += Y[j]
    for i in range(len(X)):
        result += X[i]*Ysum
    return result
```

还不错，我们去掉了内部循环，这意味着我们将 O(n²) 复杂度转换为 O(n) 复杂度。使用相同的方法，我们现在可以编写：

```python
def compute_numpy_better_3(x, y):
    Ysum = 0
    for j in range(len(Y)):
        Ysum += Y[j]
    Xsum = 0
    for i in range(len(X)):
        Xsum += X[i]
    return Xsum*Ysum
```

最后，意识到我们只需要分别对 X 和 Y 的和的乘积，我们可以利用 `np.sum` 函数并编写：

```python
def compute_numpy_better(x, y):
    return np.sum(y) * np.sum(x)
```

它更短、更清晰，而且快得多得多：

```python
X = np.arange(1000)
timeit("compute_numpy_better(X,X)", globals())
# 1000 loops, best of 3: 3.97208e-06 sec per loop
```

我们确实重新表述了我们的问题，利用了 ∑[ij]X[i]Y[j] = ∑[i]X[i]∑[j]Y[j] 这一事实，我们在此期间了解到有两种向量化：代码向量化和问题向量化。后者是最困难的，也是最重要的，因为在这里你可以期待速度的巨大提升。在这个简单的例子中，我们通过代码向量化获得了 150 倍的加速，但通过问题向量化，我们仅通过以不同的方式编写问题就获得了 70,000 倍的加速（尽管你不能在所有情况下都期待如此巨大的加速）。然而，代码向量化仍然是一个重要因素，如果我们以 Python 方式重写最后一个解决方案，改进很好，但不如 numpy 版本多：

```python
def compute_python_better(x, y):
    return sum(x)*sum(y)
```

这个新的 Python 版本比以前的 Python 版本快得多，但仍然比 numpy 版本慢 50 倍：

```python
X = np.arange(1000)
timeit("compute_python_better(X,X)", globals())
# 1000 loops, best of 3: 0.000155677 sec per loop
```

## 寻路 (Path finding)

寻路就是要在图中找到最短路径。这可以分为两个不同的问题：在图中找到两个节点之间的路径和找到最短路径。我们将通过迷宫中的寻路来说明这一点。因此，第一个任务是建立一个迷宫。

![](https://www.labri.fr/perso/nrougier/from-python-to-numpy/data/Longleat-maze-cropped.jpg)

### 构建迷宫 (Building a maze)

存在许多迷宫生成算法，但我倾向于使用我已经使用了好几年的那个，但我不知道它的起源。我已将代码添加到引用的维基百科条目中。如果您知道原作者，请随时补充。该算法通过创建 p（复杂性）个长度为 n（密度）的岛屿来工作。通过选择具有奇数坐标的随机起点，然后选择随机方向来创建岛屿。如果在给定方向上两步的单元格是空闲的，则在该方向的一步和两步处都添加墙。对于该岛屿，此过程迭代 n 步。创建 p 个岛屿。n 和 p 表示为浮点数以使其适应迷宫的大小。由于复杂性低，岛屿非常小，迷宫很容易解决。由于密度低，迷宫有更多“大空房间”。

```python
def build_maze(shape=(65, 65), complexity=0.75, density=0.50):
    # Only odd shapes
    shape = ((shape[0]//2)*2+1, (shape[1]//2)*2+1)

    # Adjust complexity and density relatively to maze size
    n_complexity = int(complexity*(shape[0]+shape[1]))
    n_density = int(density*(shape[0]*shape[1]))

    # Build actual maze
    Z = np.zeros(shape, dtype=bool)

    # Fill borders
    Z[0, :] = Z[-1, :] = Z[:, 0] = Z[:, -1] = 1

    # Islands starting point with a bias in favor of border
    P = np.random.normal(0, 0.5, (n_density, 2))
    P = 0.5 - np.maximum(-0.5, np.minimum(P, +0.5))
    P = (P*[shape[1], shape[0]]).astype(int)
    P = 2*(P//2)

    # Create islands
    for i in range(n_density):
        # Test for early stop: if all starting point are busy, this means we
        # won't be able to connect any island, so we stop.
        T = Z[2:-2:2, 2:-2:2]
        if T.sum() == T.size: break
        x, y = P[i]
        Z[y, x] = 1
        for j in range(n_complexity):
            neighbours = []
            if x > 1:          neighbours.append([(y, x-1), (y, x-2)])
            if x < shape[1]-2: neighbours.append([(y, x+1), (y, x+2)])
            if y > 1:          neighbours.append([(y-1, x), (y-2, x)])
            if y < shape[0]-2: neighbours.append([(y+1, x), (y+2, x)])
            if len(neighbours):
                choice = np.random.randint(len(neighbours))
                next_1, next_2 = neighbours[choice]
                if Z[next_2] == 0:
                    Z[next_1] = 1
                    Z[next_2] = 1
                    y, x = next_2
            else:
                break
    return Z
```

### 广度优先 (Breadth-first)

广度优先（以及深度优先）搜索算法解决了在图中找到两个节点之间路径的问题，通过从根节点开始检查所有可能性，并在找到解决方案（到达目标节点）后立即停止。该算法以线性时间运行，复杂度为 O(|V| + |E|)（其中 V 是顶点数，E 是边数）。编写这样的算法并不是特别困难，只要你有正确的数据结构。在我们的案例中，迷宫的数组表示不是最合适的，我们需要将其转换为实际的图，正如 Valentin Bryukhanov 所建议的那样。

```python
def build_graph(maze):
    height, width = maze.shape
    graph = {(i, j): [] for j in range(width)
                        for i in range(height) if not maze[i][j]}
    for row, col in graph.keys():
        if row < height - 1 and not maze[row + 1][col]:
            graph[(row, col)].append(("S", (row + 1, col)))
            graph[(row + 1, col)].append(("N", (row, col)))
        if col < width - 1 and not maze[row][col + 1]:
            graph[(row, col)].append(("E", (row, col + 1)))
            graph[(row, col + 1)].append(("W", (row, col)))
    return graph
```

> **注意**
>
> 如果我们使用深度优先算法，则不能保证找到最短路径，只能保证找到一条路径（如果存在）。

一旦完成，编写广度优先算法就很简单了。我们从起始节点开始，仅访问当前深度的节点（广度优先，记得吗？），然后迭代该过程直到到达最终节点（如果可能）。那么问题来了：以这种方式探索图能得到最短路径吗？在这个特定案例中，“是的”，因为我们没有边加权图，即所有边的权重（或成本）都相同。

```python
from collections import deque

def breadth_first(maze, start, goal):
    queue = deque([([start], start)])
    visited = set()
    graph = build_graph(maze)

    while queue:
        path, current = queue.popleft()
        if current == goal:
            return np.array(path)
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            p = list(path)
            p.append(neighbour)
            queue.append((p, neighbour))
    return None
```

### Bellman-Ford 方法 (Bellman-Ford method)

Bellman–Ford 算法是一种能够使用扩散过程在图中找到最佳路径的算法。通过提升结果梯度找到最佳路径。该算法以二次时间 O(|V||E|) 运行。但是，在我们简单的案例中，我们不会遇到最坏情况。该算法如下所示（从左到右，从上到下阅读）。一旦完成，我们可以从起始节点提升梯度。您可以在图上检查这是否会导致最短路径。

![](https://www.labri.fr/perso/nrougier/from-python-to-numpy/data/value-iteration-10.png)

我们首先将出口节点设置为值 1，而其他每个节点都设置为 0，除了墙。然后我们迭代一个过程，使得每个单元格的新值计算为当前单元格值与（gamma=0.9 在下面的情况下）4 个邻居值之间的最大值。一旦起始节点值变为严格正值，该过程就会开始。

如果我们利用 `generic_filter` (来自 `scipy.ndimage`) 进行扩散过程，numpy 实现就很简单：

```python
from scipy.ndimage import generic_filter

def diffuse(Z):
    # North, West, Center, East, South
    return max(gamma*Z[0], gamma*Z[1], Z[2], gamma*Z[3], gamma*Z[4])

# Build gradient array
G = np.zeros(Z.shape)

# Initialize gradient at the entrance with value 1
G[start] = 1

# Discount factor
gamma = 0.99

# We iterate until value at exit is > 0. This requires the maze
# to have a solution or it will be stuck in the loop.
while G[goal] == 0.0:
    G = Z * generic_filter(G, diffuse, footprint=[[0, 1, 0],
                                                  [1, 1, 1],
                                                  [0, 1, 0]])
```

但在这种特定情况下，它相当慢。我们最好自己编写解决方案，重用部分生命游戏代码：

```python
# Build gradient array
G = np.zeros(Z.shape)

# Initialize gradient at the entrance with value 1
G[start] = 1

# Discount factor
gamma = 0.99

# We iterate until value at exit is > 0. This requires the maze
# to have a solution or it will be stuck in the loop.
G_gamma = np.empty_like(G)
while G[goal] == 0.0:
    np.multiply(G, gamma, out=G_gamma)
    N = G_gamma[0:-2,1:-1]
    W = G_gamma[1:-1,0:-2]
    C = G[1:-1,1:-1]
    E = G_gamma[1:-1,2:]
    S = G_gamma[2:,1:-1]
    G[1:-1,1:-1] = Z[1:-1,1:-1]*np.maximum(N,np.maximum(W,
                                np.maximum(C,np.maximum(E,S))))
```

一旦完成，我们可以提升梯度以找到最短路径，如下图所示：

![](https://www.labri.fr/perso/nrougier/from-python-to-numpy/data/maze.png)

### 来源 (Sources)

* [maze_build.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/maze_build.py)
* [maze_numpy.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/maze_numpy.py)

## 流体动力学 (Fluid Dynamics)

### 拉格朗日与欧拉方法 (Lagrangian vs Eulerian method)

> **注意**
>
> 摘自维基百科关于[拉格朗日和欧拉规范](https://en.wikipedia.org/wiki/Lagrangian_and_Eulerian_specification_of_the_flow_field)的条目

在经典场论中，场的拉格朗日规范是一种观察流体运动的方法，观察者跟随单个流体包在空间和时间中移动。绘制单个包随时间的位置给出了包的路径线。这可以想象成坐在船上顺流而下。

流场的欧拉规范是一种观察流体运动的方法，它关注随着时间流逝流体流经的空间中的特定位置。这可以想象成坐在河岸上看着水流过固定位置。

换句话说，在欧拉情况下，您将一部分空间划分为单元格，每个单元格包含一个速度向量和其他信息，如密度和温度。在拉格朗日情况下，我们需要具有动态相互作用的基于粒子的物理学，通常我们需要大量的粒子。这两种方法各有优缺点，在两种方法之间的选择取决于您的问题性质。当然，您也可以将这两种方法混合成混合方法。

然而，基于粒子的模拟的最大问题是粒子相互作用需要找到相邻粒子，正如我们在 boids 案例中看到的那样，这是有成本的。如果我们只针对 Python 和 numpy，可能最好选择欧拉方法，因为与拉格朗日方法相比，向量化几乎是微不足道的。

### Numpy 实现 (Numpy implementation)

我不会解释计算流体动力学背后的所有理论，因为首先，我不能（我根本不是该领域的专家），而且网上有很多资源可以很好地解释这一点（看看下面的参考文献，尤其是 L. Barba 的教程）。那么为什么要选择计算流体作为例子呢？因为结果（几乎）总是美丽而迷人的。我无法抗拒（看看下面的电影）。

我们将通过实施计算机图形学中的一种方法来进一步简化问题，该方法的目标不是正确性而是令人信服的行为。Jos Stam 为 SIGGRAPH 1999 写了一篇非常好的文章，描述了一种随时间推移具有稳定流体（即长期解不会发散）的技术。Alberto Santini 很久以前就写了一个 Python 复制品（使用 numarray！），所以我只需要将其调整为现代 numpy 并使用现代 numpy 技巧稍微加速一下。

我不会评论代码，因为它太长了，但您可以阅读原始论文以及 Philip Rideout 在他的博客上的解释。

### 来源 (Sources)

* [smoke_1.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/smoke_1.py)
* [smoke_2.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/smoke_2.py)
* [smoke_solver.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/smoke_solver.py)
* [smoke_interactive.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/smoke_interactive.py)

## 蓝噪声采样 (Blue noise sampling)

蓝噪声是指具有随机且均匀分布且没有任何光谱偏差的样本集。这种噪声在各种图形应用中非常有用，如渲染、抖动、点画等。已经提出了许多不同的方法来实现这种噪声，但最简单的肯定是 DART 方法。

### DART 方法 (DART method)

DART 方法是最早和最简单的方法之一。它的工作原理是顺序绘制均匀随机点，只接受那些与每个先前接受的样本保持最小距离的点。这种顺序方法因此非常慢，因为每个新候选者都需要与先前接受的候选者进行测试。你接受的点越多，方法就越慢。让我们考虑单位表面和要在每个点之间强制执行的最小半径 r。

考虑到平面中圆的最密堆积是蜜蜂蜂窝的六边形晶格，我们知道这个密度是 d = (1)/(6)π√(3)（实际上我在写这本书时学到了它）。考虑到半径为 r 的圆，我们最多可以在表面上堆积 (d)/(πr²) = (√(3))/(6r²) = (1)/(2r²√(3)) 个。我们知道我们可以在表面上堆积的圆盘数量的理论上限，但由于随机放置，我们可能无法达到这个上限。此外，由于在接受了一些点之后很多点会被拒绝，我们需要在停止整个过程之前对连续失败试验的次数设置限制。

```python
import math
import random

def DART_sampling(width=1.0, height=1.0, r = 0.025, k=100):
    def distance(p0, p1):
        dx, dy = p0[0]-p1[0], p0[1]-p1[1]
        return math.hypot(dx, dy)

    points = []
    i = 0
    last_success = 0
    while True:
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        accept = True
        for p in points:
            if distance(p, (x, y)) < r:
                accept = False
                break
        if accept is True:
            points.append((x, y))
            if i-last_success > k:
                break
            last_success = i
        i += 1
    return points
```

我将 DART 方法的向量化留作练习。这个想法是预先计算足够的均匀随机样本以及成对距离，并测试它们的顺序包含。

### Bridson 方法 (Bridson method)

如果前一种方法的向量化没有真正的困难，速度提升也不是那么好，而且质量仍然很低，并且取决于 k 参数。越高越好，因为它基本上控制了插入新样本的努力程度。但是，当已经有大量的已接受样本时，只有运气才能让我们找到插入新样本的位置。我们可以增加 k 值，但这会使方法更慢，而且质量没有任何保证。是时候跳出框框思考了，幸运的是，Robert Bridson 为我们做到了这一点，并提出了一个简单而有效的方法：

* **步骤 0**。初始化一个 n 维背景网格以存储样本并加速空间搜索。我们选择单元格大小为 (r)/(√(n)) 为界，这样每个网格单元格最多包含一个样本，因此网格可以实现为一个简单的 n 维整数数组：默认 -1 表示没有样本，非负整数给出了位于单元格中的样本的索引。

* **步骤 1**。选择初始样本，x[0]，从域中随机均匀选择。将其插入背景网格，并用此索引（零）初始化“活动列表”（样本索引数组）。

* **步骤 2**。当活动列表不为空时，从中随机选择一个索引（比如 i）。在 x[i] 周围半径 r 和 2r 之间的球环中均匀选择最多 k 个点。对于每个点，检查它是否在现有样本的距离 r 内（使用背景网格仅测试附近的样本）。如果一个点距离现有样本足够远，将其作为下一个样本发出并将其添加到活动列表中。如果在 k 次尝试后未找到此类点，则从活动列表中删除 i。

实施没有真正的问题，留给读者作为练习。请注意，这种方法不仅速度快，而且即使使用高 k 参数，它也比 DART 方法提供更好的质量（更多样本）。

![](https://www.labri.fr/perso/nrougier/from-python-to-numpy/data/sampling.png)

### 来源 (Sources)

* [DART_sampling_python.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/DART_sampling_python.py)
* [DART_sampling_numpy.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/DART_sampling_numpy.py) (练习的解决方案)
* [Bridson_sampling.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/Bridson_sampling.py) (练习的解决方案)
* [sampling.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/sampling.py)
* [mosaic.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/mosaic.py)
* [voronoi.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/voronoi.py)

## 结论 (Conclusion)

我们一直在研究的最后一个例子确实是一个很好的例子，其中向量化问题比向量化代码更重要（而且太早）。在这个特定的案例中，我们很幸运地完成了工作，但情况并非总是如此，在这种情况下，向量化我们找到的第一个解决方案的诱惑可能会很高。我希望你现在确信，一旦你找到了一个解决方案，寻找替代解决方案通常是一个好主意。通过向量化代码，您（几乎）总是可以提高速度，但在此过程中，您可能会错过巨大的改进。

---
[返回目录](README.md)
