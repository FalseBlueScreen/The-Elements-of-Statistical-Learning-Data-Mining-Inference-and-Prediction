# 代码向量化 (Code vectorization)

## 简介 (Introduction)

代码向量化意味着您要解决的问题本质上是可向量化的，只需要一些 numpy 技巧就能让它变得更快。当然，这并不意味着它简单或直接，但至少它不需要完全重新思考您的问题（正如我们在[问题向量化](04-problem-vectorization.md)一章中将看到的那样）。不过，它可能需要一些经验才能看出哪里可以向量化代码。让我们通过一个简单的例子来说明这一点，我们要对两个整数列表求和。使用纯 Python 的一种简单方法是：

```python
def add_python(Z1,Z2):
    return [z1+z2 for (z1,z2) in zip(Z1,Z2)]
```

这个第一个朴素的解决方案可以很容易地使用 numpy 进行向量化：

```python
def add_numpy(Z1,Z2):
    return np.add(Z1,Z2)
```

毫无意外，对这两种方法进行基准测试表明第二种方法快了一个数量级。

```python
import random
from tools import timeit
Z1 = random.sample(range(1000), 100)
Z2 = random.sample(range(1000), 100)
timeit("add_python(Z1, Z2)", globals())
# 1000 loops, best of 3: 68 usec per loop
timeit("add_numpy(Z1, Z2)", globals())
# 10000 loops, best of 3: 1.14 usec per loop
```

第二种方法不仅更快，而且自然适应 Z1 和 Z2 的形状。这就是我们没有写 `Z1 + Z2` 的原因，因为如果 Z1 和 Z2 都是列表，那就行不通了。在第一个 Python 方法中，内部 `+` 的解释取决于两个对象的性质，如果我们要考虑两个嵌套列表，我们会得到以下输出：

```python
Z1 = [[1, 2], [3, 4]]
Z2 = [[5, 6], [7, 8]]
print(Z1 + Z2)
# [[1, 2], [3, 4], [5, 6], [7, 8]]
print(add_python(Z1, Z2))
# [[1, 2, 5, 6], [3, 4, 7, 8]]
print(add_numpy(Z1, Z2))
# [[ 6  8]
#  [10 12]]
```

第一种方法将两个列表连接在一起，第二种方法将内部列表连接在一起，最后一种方法计算（数值上）预期的结果。作为一个练习，你可以重写 Python 版本，使其接受任何深度的嵌套列表。

## 均匀向量化 (Uniform vectorization)

均匀向量化是最简单的向量化形式，其中所有元素在每个时间步都共享相同的计算，没有任何针对特定元素的特殊处理。一个典型的例子是约翰·康威发明的生命游戏（见下文），它是细胞自动机的最早例子之一。这些细胞自动机可以方便地视为连接在一起的细胞阵列，具有邻居的概念，其向量化是直接的。让我先定义这个游戏，然后我们看看如何向量化它。

### 生命游戏 (The Game of Life)

> **注意**
>
> 摘自维基百科关于[生命游戏](https://en.wikipedia.org/wiki/Conway's_Game_of_Life)的条目。

生命游戏是英国数学家约翰·霍顿·康威在 1970 年设计的一种细胞自动机。它是最著名的细胞自动机例子。“游戏”实际上是一个零玩家游戏，这意味着它的演变取决于它的初始状态，不需要人类玩家的输入。人们通过创建初始配置并观察其演变来与生命游戏互动。

生命游戏的宇宙是一个无限的二维正方形细胞正交网格，每个细胞都处于两种可能的状态之一：生或死。每个细胞与其八个邻居相互作用，这八个邻居是水平、垂直或对角相邻的细胞。在每个时间步，都会发生以下转换：
1. 任何少于两个活邻居的活细胞都会死亡，就像人口不足导致的那样。
2. 任何有三个以上活邻居的活细胞都会死亡，就像过度拥挤导致的那样。
3. 任何有两个或三个活邻居的活细胞都会存活到下一代，保持不变。
4. 任何恰好有三个活邻居的死细胞都会变成活细胞。

初始模式构成了系统的“种子”。第一代是通过同时将上述规则应用于种子中的每个细胞而创建的——出生和死亡同时发生，这个发生的离散时刻有时被称为滴答声。（换句话说，每一代都是前一代的纯函数。）规则继续反复应用以创建更多的代。

### Python 实现 (Python implementation)

> **注意**
>
> 我们可以使用更高效的 python `array` 接口，但使用熟悉的 `list` 对象更方便。

在纯 Python 中，我们可以使用列表的列表来编写生命游戏，代表细胞应该演变的棋盘。这样的棋盘将配备 0 的边界，这可以通过避免在计算邻居数量时对边界进行特定测试来加速事情。

```python
Z = [[0,0,0,0,0,0],
     [0,0,0,1,0,0],
     [0,1,0,1,0,0],
     [0,0,1,1,0,0],
     [0,0,0,0,0,0],
     [0,0,0,0,0,0]]
```

考虑到边界，计算邻居就很简单了：

```python
def compute_neighbours(Z):
    shape = len(Z), len(Z[0])
    N  = [[0,]*(shape[0]) for i in range(shape[1])]
    for x in range(1,shape[0]-1):
        for y in range(1,shape[1]-1):
            N[x][y] = Z[x-1][y-1]+Z[x][y-1]+Z[x+1][y-1] \
                    + Z[x-1][y]            +Z[x+1][y]   \
                    + Z[x-1][y+1]+Z[x][y+1]+Z[x+1][y+1]
    return N
```

为了迭代一步，我们只需计算每个内部细胞的邻居数量，并根据上述四个规则更新整个棋盘：

```python
def iterate(Z):
    shape = len(Z), len(Z[0])
    N = compute_neighbours(Z)
    for x in range(1,shape[0]-1):
        for y in range(1,shape[1]-1):
             if Z[x][y] == 1 and (N[x][y] < 2 or N[x][y] > 3):
                 Z[x][y] = 0
             elif Z[x][y] == 0 and N[x][y] == 3:
                 Z[x][y] = 1
    return Z
```

### Numpy 实现 (Numpy implementation)

从 Python 版本开始，生命游戏的向量化需要两部分，一部分负责计算邻居，一部分负责执行规则。如果我们记得我们在竞技场周围添加了一个空边界，计算邻居相对容易。通过考虑竞技场的部分视图，我们实际上可以非常直观地访问邻居，如下图所示的一维情况：

```
               ┏━━━┳━━━┳━━━┓───┬───┐
        Z[:-2] ┃ 0 ┃ 1 ┃ 1 ┃ 1 │ 0 │ (left neighbours)
               ┗━━━┻━━━┻━━━┛───┴───┘
                     ↓︎
           ┌───┏━━━┳━━━┳━━━┓───┐
   Z[1:-1] │ 0 ┃ 1 ┃ 1 ┃ 1 ┃ 0 │ (actual cells)
           └───┗━━━┻━━━┻━━━┛───┘
                     ↑
       ┌───┬───┏━━━┳━━━┳━━━┓
Z[+2:] │ 0 │ 1 ┃ 1 ┃ 1 ┃ 0 ┃ (right neighbours)
       └───┴───┗━━━┻━━━┻━━━┛
```

到二维情况只需要一点算术来确保考虑所有八个邻居。

```python
N = np.zeros(Z.shape, dtype=int)
N[1:-1,1:-1] += (Z[ :-2, :-2] + Z[ :-2,1:-1] + Z[ :-2,2:] +
                 Z[1:-1, :-2]                + Z[1:-1,2:] +
                 Z[2:  , :-2] + Z[2:  ,1:-1] + Z[2:  ,2:])
```

对于规则执行，我们可以使用 numpy 的 `argwhere` 方法编写第一个版本，该方法将给出给定条件为 True 的索引。

```python
# Flatten arrays
N_ = N.ravel()
Z_ = Z.ravel()

# Apply rules
R1 = np.argwhere( (Z_==1) & (N_ < 2) )
R2 = np.argwhere( (Z_==1) & (N_ > 3) )
R3 = np.argwhere( (Z_==1) & ((N_==2) | (N_==3)) )
R4 = np.argwhere( (Z_==0) & (N_==3) )

# Set new values
Z_[R1] = 0
Z_[R2] = 0
Z_[R3] = Z_[R3]
Z_[R4] = 1

# Make sure borders stay null
Z[0,:] = Z[-1,:] = Z[:,0] = Z[:,-1] = 0
```

即使这第一个版本不使用嵌套循环，它也远非最佳，因为使用了四个 `argwhere` 调用，这可能会很慢。我们可以将规则分解为将生存（保持为 1）的细胞和将出生的细胞。为此，我们可以利用 Numpy 布尔能力并非常自然地编写：

> **注意**
>
> 我们没有写 `Z = 0`，因为这会简单地将值 0 分配给 Z，然后 Z 将变成一个简单的标量。

```python
birth = (N==3)[1:-1,1:-1] & (Z[1:-1,1:-1]==0)
survive = ((N==2) | (N==3))[1:-1,1:-1] & (Z[1:-1,1:-1]==1)
Z[...] = 0
Z[1:-1,1:-1][birth | survive] = 1
```

如果您查看 birth 和 survive 行，您会看到这两个变量是数组，可用于在清除 Z 后将 Z 值设置为 1。

### 练习 (Exercise)

化学物质的反应和扩散可以产生各种模式，让人想起自然界中经常看到的模式。Gray-Scott 方程模拟了这种反应。有关此化学系统的更多信息，请参阅文章 Complex Patterns in a Simple System (John E. Pearson, Science, Volume 261, 1993)。让我们考虑两种化学物质 U 和 V，其浓度分别为 u 和 v，扩散率为 Du 和 Dv。V 以转化率 k 转化为 P。f 代表供给 U 并排出 U、V 和 P 的过程的速率。这可以写成：

| 化学反应 | 方程 |
|---|---|
| U + 2V → 3V | u̇ = Du∇²u − uv² + f(1 − u) |
| V → P | v̇ = Dv∇²v + uv² − (f + k)v |

基于生命游戏的例子，尝试实现这样的反应扩散系统。这是一组要测试的有趣参数：

| Name | Du | Dv | f | k |
|---|---|---|---|---|
| Bacteria 1 | 0.16 | 0.08 | 0.035 | 0.065 |
| Bacteria 2 | 0.14 | 0.06 | 0.035 | 0.065 |
| Coral | 0.16 | 0.08 | 0.060 | 0.062 |
| Fingerprint | 0.19 | 0.05 | 0.060 | 0.062 |
| Spirals | 0.10 | 0.10 | 0.018 | 0.050 |
| Spirals Dense | 0.12 | 0.08 | 0.020 | 0.050 |
| Spirals Fast | 0.10 | 0.16 | 0.020 | 0.050 |
| Unstable | 0.16 | 0.08 | 0.020 | 0.055 |
| Worms 1 | 0.16 | 0.08 | 0.050 | 0.065 |
| Worms 2 | 0.16 | 0.08 | 0.054 | 0.063 |
| Zebrafish | 0.16 | 0.08 | 0.035 | 0.060 |

### 来源 (Sources)

* [game_of_life_python.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/game_of_life_python.py)
* [game_of_life_numpy.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/game_of_life_numpy.py)
* [gray_scott.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/gray_scott.py) (练习的解决方案)

## 时间向量化 (Temporal vectorization)

Mandelbrot 集是复数 c 的集合，对于该集合，函数 f_c(z) = z² + c 从 z = 0 开始迭代时不会发散，即序列 f_c(0), f_c(f_c(0)) 等的绝对值保持有界。它很容易计算，但可能需要很长时间，因为您需要确保给定数字不发散。这通常通过将计算迭代到最大迭代次数来完成，之后，如果数字仍在某些界限内，则认为它是非发散的。当然，迭代次数越多，精度就越高。

### Python 实现 (Python implementation)

纯 python 实现写为：

```python
def mandelbrot_python(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    def mandelbrot(z, maxiter):
        c = z
        for n in range(maxiter):
            if abs(z) > horizon:
                return n
            z = z*z + c
        return maxiter
    r1 = [xmin+i*(xmax-xmin)/xn for i in range(xn)]
    r2 = [ymin+i*(ymax-ymin)/yn for i in range(yn)]
    return [mandelbrot(complex(r, i),maxiter) for r in r1 for i in r2]
```

这段代码中有趣（且慢）的部分是实际计算序列 f_c(f_c(f_c...))) 的 `mandelbrot` 函数。这种代码的向量化并不完全直接，因为内部返回意味着对元素的差异处理。一旦发散，我们就不需要再迭代了，我们可以安全地返回发散时的迭代计数。问题是如何在 numpy 中做同样的事情。怎么做？

### Numpy 实现 (Numpy implementation)

技巧是在每次迭代中搜索尚未发散的值，并更新这些值的相关信息，且仅更新这些值。因为我们从 Z = 0 开始，我们知道每个值至少会被更新一次（当它们等于 0 时，它们尚未发散），并且一旦它们发散就会停止更新。为此，我们将使用 numpy 花式索引和 `less(x1,x2)` 函数，该函数按元素返回 (x1 < x2) 的真值。

```python
def mandelbrot_numpy(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    X = np.linspace(xmin, xmax, xn, dtype=np.float32)
    Y = np.linspace(ymin, ymax, yn, dtype=np.float32)
    C = X + Y[:,None]*1j
    N = np.zeros(C.shape, dtype=int)
    Z = np.zeros(C.shape, np.complex64)
    for n in range(maxiter):
        I = np.less(abs(Z), horizon)
        N[I] = n
        Z[I] = Z[I]**2 + C[I]
    N[N == maxiter-1] = 0
    return Z, N
```

这是基准测试：

```python
>>> xmin, xmax, xn = -2.25, +0.75, int(3000/3)
>>> ymin, ymax, yn = -1.25, +1.25, int(2500/3)
>>> maxiter = 200
>>> timeit("mandelbrot_python(xmin, xmax, ymin, ymax, xn, yn, maxiter)", globals())
1 loops, best of 3: 6.1 sec per loop
>>> timeit("mandelbrot_numpy(xmin, xmax, ymin, ymax, xn, yn, maxiter)", globals())
1 loops, best of 3: 1.15 sec per loop
```

### 更快的 numpy 实现 (Faster numpy implementation)

增益大约是 5 倍，没有我们预期的那么多。部分问题在于 `np.less` 函数意味着每次迭代都要进行 xn × yn 次测试，而我们知道有些值已经发散了。即使这些测试是在 C 级别（通过 numpy）执行的，成本仍然很大。Dan Goodman 提出的另一种方法是在每次迭代中处理一个动态数组，该数组仅存储尚未发散的点。它需要更多行代码，但结果更快，与 Python 版本相比，速度提高了 10 倍。

```python
def mandelbrot_numpy_2(xmin, xmax, ymin, ymax, xn, yn, itermax, horizon=2.0):
    Xi, Yi = np.mgrid[0:xn, 0:yn]
    Xi, Yi = Xi.astype(np.uint32), Yi.astype(np.uint32)
    X = np.linspace(xmin, xmax, xn, dtype=np.float32)[Xi]
    Y = np.linspace(ymin, ymax, yn, dtype=np.float32)[Yi]
    C = X + Y*1j
    N_ = np.zeros(C.shape, dtype=np.uint32)
    Z_ = np.zeros(C.shape, dtype=np.complex64)
    Xi.shape = Yi.shape = C.shape = xn*yn

    Z = np.zeros(C.shape, np.complex64)
    for i in range(itermax):
        if not len(Z): break

        # 仅为相关点计算
        np.multiply(Z, Z, Z)
        np.add(Z, C, Z)

        # 失败的收敛
        I = abs(Z) > horizon
        N_[Xi[I], Yi[I]] = i+1
        Z_[Xi[I], Yi[I]] = Z[I]

        # 继续处理那些尚未发散的
        np.negative(I,I)
        Z = Z[I]
        Xi, Yi = Xi[I], Yi[I]
        C = C[I]
    return Z_.T, N_.T
```

基准测试给出：

```python
>>> timeit("mandelbrot_numpy_2(xmin, xmax, ymin, ymax, xn, yn, maxiter)", globals())
1 loops, best of 3: 510 msec per loop
```

### 练习 (Exercise)

> **注意**
>
> 您应该查看 `ufunc.reduceat` 方法，该方法对单个轴执行指定切片的（局部）归约。

我们现在想使用 Minkowski–Bouligand 维度测量 Mandelbrot 集的分形维数。为此，我们需要使用减小的框大小进行盒计数。正如你可以想象的那样，我们不能使用纯 Python，因为它太慢了。练习的目标是使用 numpy 编写一个函数，该函数接受一个二维浮点数组并返回维数。我们将考虑数组中的值已归一化（即所有值都在 0 和 1 之间）。

### 来源 (Sources)

* [mandelbrot.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/mandelbrot.py)
* [mandelbrot_python.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/mandelbrot_python.py)
* [mandelbrot_numpy_1.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/mandelbrot_numpy_1.py)
* [mandelbrot_numpy_2.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/mandelbrot_numpy_2.py)
* [fractal_dimension.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/fractal_dimension.py) (练习的解决方案)

## 空间向量化 (Spatial vectorization)

空间向量化是指元素共享相同的计算，但仅与一部分其他元素相互作用的情况。生命游戏的例子已经是这种情况，但在某些情况下，还有一个额外的困难，因为子组是动态的，需要在每次迭代时更新。例如，在粒子主要与局部邻居相互作用的粒子系统中就是这种情况。模拟群集行为的 "boids" 也是这种情况。

### Boids

> **注意**
>
> 摘自维基百科条目 [Boids](https://en.wikipedia.org/wiki/Boids)

Boids 是 Craig Reynolds 在 1986 年开发的一个人工生命程序，它模拟了鸟类的群集行为。"boid" 这个名字对应于 "bird-oid object" 的缩写，指的是一种像鸟一样的物体。

与大多数人工生命模拟一样，Boids 是涌现行为的一个例子；也就是说，Boids 的复杂性来自于遵循一组简单规则的个体代理（在这种情况下是 boids）的相互作用。在最简单的 Boids 世界中应用的规则如下：
* 分离 (separation)：转向以避免拥挤局部群落伙伴
* 对齐 (alignment)：转向局部群落伙伴的平均航向
* 凝聚 (cohesion)：转向以向局部群落伙伴的平均位置（质心）移动

### Python 实现 (Python implementation)

由于每个 boid 都是具有位置和速度等多个属性的自治实体，因此从编写 Boid 类开始似乎很自然：

```python
import math
import random
from vec2 import vec2

class Boid:
    def __init__(self, x=0, y=0):
        self.position = vec2(x, y)
        angle = random.uniform(0, 2*math.pi)
        self.velocity = vec2(math.cos(angle), math.sin(angle))
        self.acceleration = vec2(0, 0)
```

`vec2` 对象是一个非常简单的类，它处理具有 2 个分量的所有常见向量运算。它将为我们在主 Boid 类中节省一些编写。请注意，Python Package Index 中有一些向量包，但对于这样一个简单的例子来说，这有点大材小用。

Boid 对于常规 Python 来说是一个困难的案例，因为 boid 与局部邻居有相互作用。然而，由于 boids 在移动，为了找到这样的局部邻居，需要在每个时间步计算到每个其他 boid 的距离，以便对那些处于给定相互作用半径内的 boid 进行排序。编写这三个规则的原型方式如下：

```python
def separation(self, boids):
    count = 0
    for other in boids:
        d = (self.position - other.position).length()
        if 0 < d < desired_separation:
            count += 1
            ...
    if count > 0:
        ...

 def alignment(self, boids): ...
 def cohesion(self, boids): ...
```

完整的源代码在下面的参考文献部分给出，在这里描述太长了，而且没有真正的困难。

为了完成这幅图，我们还可以创建一个 Flock 对象：

```python
class Flock:
    def __init__(self, count=150):
        self.boids = []
        for i in range(count):
            boid = Boid()
            self.boids.append(boid)

    def run(self):
        for boid in self.boids:
            boid.run(self.boids)
```

使用这种方法，我们可以有多达 50 个 boids，直到计算时间变得太慢而无法进行流畅的动画。正如您可能猜到的那样，我们可以使用 numpy 做得更好，但让我首先指出这个 Python 实现的主要问题。如果你看代码，你肯定会注意到有很多冗余。更准确地说，我们没有利用欧几里得距离是自反的这一事实，即 `|x - y| = |y - x|`。在这个朴素的 Python 实现中，每个规则（函数）计算 n² 个距离，而如果适当地缓存，(n²)/2 就足够了。此外，每个规则都会重新计算每个距离，而不会为其他函数缓存结果。最后，我们计算了 3n² 个距离，而不是 (n²)/2。

### Numpy 实现 (Numpy implementation)

正如您可能预期的那样，numpy 实现采用了不同的方法，我们将所有的 boids 收集到一个位置数组和一个速度数组中：

```python
n = 500
velocity = np.zeros((n, 2), dtype=np.float32)
position = np.zeros((n, 2), dtype=np.float32)
```

第一步是计算所有 boids 的局部邻域，为此我们需要计算所有配对距离：

```python
dx = np.subtract.outer(position[:, 0], position[:, 0])
dy = np.subtract.outer(position[:, 1], position[:, 1])
distance = np.hypot(dx, dy)
```

我们可以使用 scipy 的 `cdist`，但我们稍后需要 `dx` 和 `dy` 数组。一旦计算出这些，使用 `hypot` 方法会更快。请注意，距离形状为 (n, n)，每一行都与一个 boid 相关，即每一行给出了到所有其他 boids（包括自身）的距离。

根据这些距离，我们现在可以为这三个规则中的每一个计算局部邻域，利用我们可以将它们混合在一起的事实。我们实际上可以计算严格正距离（即没有自相互作用）的掩码，并将其与其他距离掩码相乘。

> **注意**
>
> 如果我们假设 boids 不能占据相同的位置，你如何更有效地计算 mask_0？

```python
mask_0 = (distance > 0)
mask_1 = (distance < 25)
mask_2 = (distance < 50)
mask_1 *= mask_0
mask_2 *= mask_0
mask_3 = mask_2
```

然后，我们计算给定半径内的邻居数量，并确保它至少为 1 以避免除以零。

```python
mask_1_count = np.maximum(mask_1.sum(axis=1), 1)
mask_2_count = np.maximum(mask_2.sum(axis=1), 1)
mask_3_count = mask_2_count
```

我们准备好编写我们的三个规则了：

**对齐 (Alignment)**

```python
# 计算局部邻居的平均速度
target = np.dot(mask, velocity)/count.reshape(n, 1)

# 归一化结果
norm = np.sqrt((target*target).sum(axis=1)).reshape(n, 1)
target *= np.divide(target, norm, out=target, where=norm != 0)

# 恒定速度对齐
target *= max_velocity

# 计算产生的转向
alignment = target - velocity
```

**凝聚 (Cohesion)**

```python
# 计算局部邻居的重心
center = np.dot(mask, position)/count.reshape(n, 1)

# 计算朝向中心的方向
target = center - position

# 归一化结果
norm = np.sqrt((target*target).sum(axis=1)).reshape(n, 1)
target *= np.divide(target, norm, out=target, where=norm != 0)

# 恒定速度凝聚 (max_velocity)
target *= max_velocity

# 计算产生的转向
cohesion = target - velocity
```

**分离 (Separation)**

```python
# 计算来自局部邻居的排斥力
repulsion = np.dstack((dx, dy))

# 力与距离成反比
repulsion = np.divide(repulsion, distance.reshape(n, n, 1)**2, out=repulsion,
                      where=distance.reshape(n, n, 1) != 0)

# 计算远离他人的方向
target = (repulsion*mask.reshape(n, n, 1)).sum(axis=1)/count.reshape(n, 1)

# 归一化结果
norm = np.sqrt((target*target).sum(axis=1)).reshape(n, 1)
target *= np.divide(target, norm, out=target, where=norm != 0)

# 恒定速度分离 (max_velocity)
target *= max_velocity

# 计算产生的转向
separation = target - velocity
```

所有三个产生的转向（分离、对齐和凝聚）都需要在幅度上受到限制。我们将此作为读者的练习。这些规则的组合很简单，以及由此产生的速度和位置更新：

```python
acceleration = 1.5 * separation + alignment + cohesion
velocity += acceleration
position += velocity
```

我们最终使用自定义定向散点图可视化结果。

### 练习 (Exercise)

我们现在准备好可视化我们的 boids 了。最简单的方法是使用 matplotlib 动画函数和散点图。不幸的是，散点图不能单独定向，我们需要使用 matplotlib PathCollection 制作我们自己的对象。简单的三角形路径可以定义为：

```python
v= np.array([(-0.25, -0.25),
             ( 0.00,  0.50),
             ( 0.25, -0.25),
             ( 0.00,  0.00)])
c = np.array([Path.MOVETO,
              Path.LINETO,
              Path.LINETO,
              Path.CLOSEPOLY])
```

此路径可以在数组内重复多次，并且每个三角形都可以独立制作。

```python
n = 500
vertices = np.tile(v.reshape(-1), n).reshape(n, len(v), 2)
codes = np.tile(c.reshape(-1), n)
```

我们现在有一个 (n,4,2) 的顶点数组和一个 (n,4) 的代码数组，代表 n 个 boids。我们有兴趣操纵顶点数组以反映每个 boid 的平移、缩放和旋转。

> **注意**
>
> 旋转真的很棘手。

你会怎么写平移、缩放和旋转函数？

### 来源 (Sources)

* [boid_python.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/boid_python.py)
* [boid_numpy.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/boid_numpy.py) (练习的解决方案)

## 结论 (Conclusion)

我们通过这些例子看到了三种形式的代码向量化：
* **均匀向量化**，其中元素无条件地且在相同的持续时间内共享相同的计算。
* **时间向量化**，其中元素共享相同的计算，但需要不同数量的迭代。
* **空间向量化**，其中元素共享相同的计算，但针对动态空间参数。

而且可能有更多形式的这种直接代码向量化。如前所述，这种向量化是最简单的一种，尽管我们已经看到它实施起来可能真的很棘手，需要一些经验、一些帮助或两者兼而有之。例如，boids 练习的解决方案由 [Divakar](http://stackoverflow.com/users/3293881/divakar) 在 [stack overflow](http://stackoverflow.com/questions/40822983/multiple-individual-2d-rotation-at-once) 上在我解释了我的问题后提供。

---
[返回目录](README.md)
