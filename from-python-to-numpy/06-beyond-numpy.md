# 超越 Numpy (Beyond Numpy)

## 回到 Python (Back to Python)

您几乎已经到了本书的结尾，希望您已经了解到 numpy 是一个非常通用和强大的库。然而，与此同时，请记住 Python 也是一种非常强大的语言。事实上，在某些特定情况下，它可能比 numpy 更强大。让我们考虑例如 Tucker Balch 在他的 [Coursera 计算投资课程](https://www.coursera.org/learn/computational-investing)中提出的一个有趣练习。练习写为：

> 编写最简洁的代码，计算分配给 4 只股票的所有 "合法" 分配，使得分配以 1.0 为块，并且分配总和为 10.0。

[Yaser Martinez](http://yasermartinez.com/blog/index.html) 收集了来自社区的不同答案，提出的解决方案产生了令人惊讶的结果。但让我们从最明显的 Python 解决方案开始：

```python
def solution_1():
    # 蛮力法
    # 14641 (=11*11*11*11) 次迭代和测试
    Z = []
    for i in range(11):
        for j in range(11):
            for k in range(11):
                for l in range(11):
                    if i+j+k+l == 10:
                        Z.append((i,j,k,l))
    return Z
```

这个解决方案是最慢的，因为它需要 4 个循环，更重要的是，它测试了 0 到 10 之间 4 个整数的所有不同组合 (14641) 以仅保留总和为 10 的组合。我们当然可以使用 `itertools` 摆脱这 4 个循环，但代码仍然很慢：

```python
import itertools as it

def solution_2():
    # Itertools
    # 14641 (=11*11*11*11) 次迭代和测试
    return [(i,j,k,l)
            for i,j,k,l in it.product(range(11),repeat=4) if i+j+k+l == 10]
```

Nick Popplas 提出的最好的解决方案之一利用了我们可以拥有智能嵌套循环的事实，这将允许我们直接构建每个元组而无需任何测试，如下所示：

```python
def solution_3():
    return [(a, b, c, (10 - a - b - c))
            for a in range(11)
            for b in range(11 - a)
            for c in range(11 - a - b)]
```

Yaser Martinez 提出的最佳 numpy 解决方案使用了具有一组受限测试的不同策略：

```python
def solution_4():
    X123 = np.indices((11,11,11)).reshape(3,11*11*11)
    X4 = 10 - X123.sum(axis=0)
    return np.vstack((X123, X4)).T[X4 > -1]
```

如果我们对这些方法进行基准测试，我们会得到：

```python
>>> timeit("solution_1()", globals())
100 loops, best of 3: 1.9 msec per loop
>>> timeit("solution_2()", globals())
100 loops, best of 3: 1.67 msec per loop
>>> timeit("solution_3()", globals())
1000 loops, best of 3: 60.4 usec per loop
>>> timeit("solution_4()", globals())
1000 loops, best of 3: 54.4 usec per loop
```

Numpy 解决方案最快，但纯 Python 解决方案具有可比性。但让我对 Python 解决方案做一个小修改：

```python
def solution_3_bis():
    return ((a, b, c, (10 - a - b - c))
            for a in range(11)
            for b in range(11 - a)
            for c in range(11 - a - b))
```

如果我们对其进行基准测试，我们会得到：

```python
>>> timeit("solution_3_bis()", globals())
10000 loops, best of 3: 0.643 usec per loop
```

你没看错，我们仅通过用圆括号替换方括号就获得了 100 倍的增益。这怎么可能？通过查看返回对象的类型可以找到解释：

```python
>>> print(type(solution_3()))
<class 'list'>
>>> print(type(solution_3_bis()))
<class 'generator'>
```

`solution_3_bis()` 返回一个生成器，可用于生成完整列表或迭代所有不同元素。在任何情况下，巨大的加速来自于未实例化完整列表，因此重要的是要知道你需要结果的实际实例还是简单的生成器就可以完成工作。

## Numpy & co

除了 numpy 之外，还有其他几个 Python 包值得一看，因为它们使用不同的技术（编译、虚拟机、即时编译、GPU、压缩等）解决类似但不同类的问题。根据您的具体问题和硬件，一个包可能比另一个更好。让我们使用一个非常简单的例子来说明它们的用法，我们要计算基于两个浮点向量的表达式：

```python
import numpy as np
a = np.random.uniform(0, 1, 1000).astype(np.float32)
b = np.random.uniform(0, 1, 1000).astype(np.float32)
c = 2*a + 3*b
```

### NumExpr

[NumExpr](https://github.com/pydata/numexpr/wiki/Numexpr-Users-Guide) 包通过使用基于向量的虚拟机为数组表达式的快速求值提供例程。它类似于 SciPy 的 weave 包，但不需要单独编译 C 或 C++ 代码的步骤。

```python
import numpy as np
import numexpr as ne

a = np.random.uniform(0, 1, 1000).astype(np.float32)
b = np.random.uniform(0, 1, 1000).astype(np.float32)
c = ne.evaluate("2*a + 3*b")
```

### Cython

[Cython](http://cython.org/) 是针对 Python 编程语言和扩展 Cython 编程语言（基于 Pyrex）的优化静态编译器。它使编写 Python 的 C 扩展就像 Python 本身一样容易。

```python
import numpy as np

def evaluate(np.ndarray a, np.ndarray b):
    cdef int i
    cdef np.ndarray c = np.zeros_like(a)
    for i in range(a.size):
        c[i] = 2*a[i] + 3*b[i]
    return c

a = np.random.uniform(0, 1, 1000).astype(np.float32)
b = np.random.uniform(0, 1, 1000).astype(np.float32)
c = evaluate(a, b)
```

### Numba

[Numba](http://numba.pydata.org/) 使您能够通过直接用 Python 编写的高性能函数来加速您的应用程序。通过一些注释，面向数组和数学繁重的 Python 代码可以即时编译为本机机器指令，性能类似于 C、C++ 和 Fortran，而无需切换语言或 Python 解释器。

```python
from numba import jit
import numpy as np

@jit
def evaluate(np.ndarray a, np.ndarray b):
    c = np.zeros_like(a)
    for i in range(a.size):
        c[i] = 2*a[i] + 3*b[i]
    return c

a = np.random.uniform(0, 1, 1000).astype(np.float32)
b = np.random.uniform(0, 1, 1000).astype(np.float32)
c = evaluate(a, b)
```

### Theano

[Theano](http://www.deeplearning.net/software/theano/) 是一个 Python 库，允许您有效地定义、优化和评估涉及多维数组的数学表达式。Theano 具有与 numpy 的紧密集成、透明使用 GPU、有效的符号微分、速度和稳定性优化、动态 C 代码生成以及广泛的单元测试和自验证。

```python
import numpy as np
import theano.tensor as T

x = T.fvector('x')
y = T.fvector('y')
z = 2*x + 3*y
f = function([x, y], z)

a = np.random.uniform(0, 1, 1000).astype(np.float32)
b = np.random.uniform(0, 1, 1000).astype(np.float32)
c = f(a, b)
```

### PyCUDA

[PyCUDA](http://mathema.tician.de/software/pycuda) 允许您从 Python 访问 Nvidia 的 CUDA 并行计算 API。

```python
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

mod = SourceModule("""
    __global__ void evaluate(float *c, float *a, float *b)
    {
      const int i = threadIdx.x;
      c[i] = 2*a[i] + 3*b[i];
    }
""")

evaluate = mod.get_function("evaluate")

a = np.random.uniform(0, 1, 1000).astype(np.float32)
b = np.random.uniform(0, 1, 1000).astype(np.float32)
c = np.zeros_like(a)

evaluate(drv.Out(c), drv.In(a), drv.In(b), block=(400,1,1), grid=(1,1))
```

### PyOpenCL

[PyOpenCL](http://mathema.tician.de/software/pyopencl) 允许您从 Python 访问 GPU 和其他大规模并行计算设备。

```python
import numpy as np
import pyopencl as cl

a = np.random.uniform(0, 1, 1000).astype(np.float32)
b = np.random.uniform(0, 1, 1000).astype(np.float32)
c = np.empty_like(a)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
gpu_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
gpu_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

evaluate = cl.Program(ctx, """
    __kernel void evaluate(__global const float *gpu_a;
                           __global const float *gpu_b;
                           __global       float *gpu_c)
    {
        int gid = get_global_id(0);
        gpu_c[gid] = 2*gpu_a[gid] + 3*gpu_b[gid];
    }
""").build()

gpu_c = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)
evaluate.evaluate(queue, a.shape, None, gpu_a, gpu_b, gpu_c)
cl.enqueue_copy(queue, c, gpu_c)
```

## Scipy & co

如果 numpy 有几个附加包，那么 scipy 有一万亿个附加包。事实上，每个科学领域都可能有自己的包，我们之前研究的大多数例子都可以在相关包中通过两三个方法调用来解决。当然，那不是目标，如果你有一些空闲时间，自己编程通常是一个很好的练习。此时最大的困难是找到这些相关包。这里有一个非常简短的包列表，它们维护良好、测试充分，可能会简化您的科学生活（取决于您的领域）。当然还有更多，根据您的具体需求，您可能不必自己编写所有程序。有关广泛的列表，请查看 [Awesome python list](https://awesome-python.com/)。

### scikit-learn

[scikit-learn](http://scikit-learn.org/stable/) 是 Python 编程语言的免费软件机器学习库。它具有各种分类、回归和聚类算法，包括支持向量机、随机森林、梯度提升、k-means 和 DBSCAN，并且旨在与 Python 数值和科学库 numpy 和 SciPy 互操作。

### scikit-image

[scikit-image](http://scikit-image.org/) 是一个致力于图像处理的 Python 包，并原生使用 numpy 数组作为图像对象。本章描述了如何在各种图像处理任务上使用 scikit-image，并坚持与其他科学 Python 模块（如 numpy 和 SciPy）的链接。

### SymPy

[SymPy](http://www.sympy.org/en/index.html) 是一个用于符号数学的 Python 库。它旨在成为一个全功能的计算机代数系统 (CAS)，同时保持代码尽可能简单，以便易于理解和扩展。SymPy 完全用 Python 编写。

### Astropy

[Astropy](http://www.astropy.org/) 项目是社区努力开发的一个用于 Python 中天文学的单一核心包，并促进 Python 天文学包之间的互操作性。

### Cartopy

[Cartopy](http://scitools.org.uk/cartopy/) 是一个 Python 包，旨在使绘制数据分析和可视化地图尽可能容易。Cartopy 利用了强大的 PROJ.4、numpy 和 shapely 库，并具有简单直观的 matplotlib 绘图界面，用于创建出版质量的地图。

### Brian

[Brian](http://www.briansimulator.org/) 是一个免费、开源的脉冲神经网络模拟器。它是用 Python 编程语言编写的，几乎可以在所有平台上使用。我们相信模拟器不仅应该节省处理器的时间，还应该节省科学家的时间。因此，Brian 旨在易于学习和使用，高度灵活且易于扩展。

### Glumpy

[Glumpy](http://glumpy.github.io/) 是一个基于 Python 的 OpenGL 交互式可视化库。它的目标是使创建快速、可扩展、美观、交互式和动态的可视化变得容易。

## 结论 (Conclusion)

Numpy 是一个非常通用的库，但这并不意味着您必须在每种情况下都使用它。在本章中，我们看到了一些值得一看的替代方案（包括 Python 本身）。一如既往，选择权在您手中。您必须考虑什么是在开发时间、计算时间和维护工作方面最适合您的解决方案。一方面，如果您设计自己的解决方案，您将不得不对其进行测试和维护，但作为交换，您可以自由地按照您想要的方式设计它。另一方面，如果您决定依赖第三方包，您将节省开发时间并受益于社区支持，即使您可能不得不调整包以适应您的特定需求。选择权在您。

---
[返回目录](README.md)
