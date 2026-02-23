# 简介 (Introduction)

关于 Numpy 的书籍已经相当多了（见[参考文献](08-bibliography.md)），一个合理的问题是是否真的需要另一本书。正如您阅读这些行所猜到的那样，我个人的回答是肯定的，主要是因为我认为有空间采用一种不同的方法，专注于通过向量化从 Python 迁移到 Numpy。有很多技术是您在书中找不到的，这些技术主要通过经验学习。本书的目标是解释其中一些技术，并提供在这个过程中获得这些经验的机会。

![](https://www.labri.fr/perso/nrougier/from-python-to-numpy/data/cubes.png)

## 简单示例 (Simple example)

> **注意**
>
> 您可以使用常规 python shell 或在 IPython 会话或 Jupyter notebook 中从 code 文件夹执行下面的任何代码。在这种情况下，您可能希望使用魔术命令 `%timeit` 而不是我编写的自定义命令。

Numpy 的核心在于向量化。如果您熟悉 Python，这将是您面临的主要困难，因为您需要改变思维方式，而您的新朋友（除其他外）名为 "vectors"（向量）、"arrays"（数组）、"views"（视图）或 "ufuncs"（通用函数）。

让我们举一个非常简单的例子，随机游走。一种可能的面向对象方法是定义一个 `RandomWalker` 类，并编写一个 `walk` 方法，该方法在每一步（随机）后返回当前位置。它很好，很易读，但很慢：

**面向对象方法**

```python
import random

class RandomWalker:
    def __init__(self):
        self.position = 0

    def walk(self, n):
        self.position = 0
        for i in range(n):
            yield self.position
            self.position += 2*random.randint(0, 1) - 1

walker = RandomWalker()
walk = [position for position in walker.walk(1000)]
```

基准测试给出了：

```python
>>> from tools import timeit
>>> walker = RandomWalker()
>>> timeit("[position for position in walker.walk(n=10000)]", globals())
10 loops, best of 3: 15.7 msec per loop
```

**过程式方法**

对于这样一个简单的问题，我们可能可以省去类定义，只专注于计算每个随机步骤后连续位置的 `walk` 方法。

```python
def random_walk(n):
    position = 0
    walk = [position]
    for i in range(n):
        position += 2*random.randint(0, 1)-1
        walk.append(position)
    return walk

walk = random_walk(1000)
```

这种新方法节省了一些 CPU 周期，但不是很多，因为此函数与面向对象方法几乎相同，我们节省的少数周期可能来自内部 Python 面向对象机制。

```python
>>> from tools import timeit
>>> timeit("random_walk(n=10000)", globals())
10 loops, best of 3: 15.6 msec per loop
```

**向量化方法**

但我们可以使用 `itertools` Python 模块做得更好，该模块提供了一组用于高效循环的函数。如果我们观察到随机游走是步骤的累积，我们可以通过首先生成所有步骤然后累积它们而无需任何循环来重写函数：

```python
def random_walk_faster(n=1000):
    from itertools import accumulate
    # 仅从 Python 3.6 开始可用
    steps = random.choices([-1,+1], k=n)
    return [0]+list(accumulate(steps))

walk = random_walk_faster(1000)
```

事实上，我们刚刚向量化了我们的函数。我们没有通过循环来选择连续的步骤并将它们添加到当前位置，而是一次生成所有步骤，并使用 `accumulate` 函数计算所有位置。我们摆脱了循环，这使事情变得更快：

```python
>>> from tools import timeit
>>> timeit("random_walk_faster(n=10000)", globals())
10 loops, best of 3: 2.21 msec per loop
```

与之前的版本相比，我们获得了 85% 的计算时间增益，还不错。但是这个新版本的优势在于它使 numpy 向量化变得超级简单。我们只需将 `itertools` 调用转换为 numpy 调用。

```python
import numpy as np

def random_walk_fastest(n=1000):
    # numpy choice 中没有 's' (Python 提供 choice & choices)
    steps = np.random.choice([-1,+1], n)
    return np.cumsum(steps)

walk = random_walk_fastest(1000)
```

并不太难，但我们使用 numpy 获得了 500 倍的增益：

```python
>>> from tools import timeit
>>> timeit("random_walk_fastest(n=10000)", globals())
1000 loops, best of 3: 14 usec per loop
```

本书是关于向量化的，无论是代码层面还是问题层面。在查看自定义向量化之前，我们将看到这种差异很重要。

## 可读性与速度 (Readability vs speed)

在进入下一章之前，我想提醒您一旦熟悉 numpy 可能会遇到的潜在问题。它是一个非常强大的库，你可以用它创造奇迹，但大多数时候，这是以牺牲可读性为代价的。如果您在编写代码时没有对其进行注释，几周（甚至几天）后您将无法分辨函数在做什么。例如，您能分辨出下面两个函数在做什么吗？您可能可以说出第一个函数，但不一定能说出第二个函数（或者您的名字是 Jaime Fernández del Río 并且您不需要读这本书）。

```python
def function_1(seq, sub):
    return [i for i in range(len(seq) - len(sub)) if seq[i:i+len(sub)] == sub]

def function_2(seq, sub):
    target = np.dot(sub, sub)
    candidates = np.where(np.correlate(seq, sub, mode='valid') == target)[0]
    check = candidates[:, np.newaxis] + np.arange(len(sub))
    mask = np.all((np.take(seq, check) == sub), axis=-1)
    return candidates[mask]
```

正如您可能猜到的那样，第二个函数是第一个函数的向量化-优化-更快的 numpy 版本。它比纯 Python 版本快 10 倍，但几乎不可读。

---
[返回目录](README.md)
