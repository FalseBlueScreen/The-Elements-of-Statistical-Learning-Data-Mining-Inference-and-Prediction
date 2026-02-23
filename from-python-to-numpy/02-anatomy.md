# 数组剖析 (Anatomy of an array)

## 简介 (Introduction)

正如在[前言](00-preface.md)中所解释的，您应该具有使用 numpy 的基本经验才能阅读本书。如果不是这种情况，您最好先从初学者教程开始，然后再回到这里。因此，我将在这里仅快速回顾 numpy 数组的基本剖析，特别是关于内存布局、视图、副本和数据类型。如果您希望您的计算从 numpy 哲学中受益，这些是需要理解的关键概念。

让我们考虑一个简单的例子，我们要清除 dtype 为 `np.float32` 的数组中的所有值。如何编写它以最大化速度？下面的语法相当明显（至少对于那些熟悉 numpy 的人来说），但上述问题要求找到最快的操作。

```python
import numpy as np
Z = np.ones(4*1000000, np.float32)
Z[...] = 0
```

如果您更仔细地观察数组的 dtype 和大小，您可以观察到该数组可以被转换为（即视为）许多其他 "兼容" 的数据类型。通过兼容，我的意思是 `Z.size * Z.itemsize` 可以被新 dtype itemsize 整除。

```python
from tools import timeit
Z = np.ones(4*1000000, np.float32)
timeit("Z.view(np.float16)[...] = 0", globals())
# 100 loops, best of 3: 2.72 msec per loop
timeit("Z.view(np.int16)[...] = 0", globals())
# 100 loops, best of 3: 2.77 msec per loop
timeit("Z.view(np.int32)[...] = 0", globals())
# 100 loops, best of 3: 1.29 msec per loop
timeit("Z.view(np.float32)[...] = 0", globals())
# 100 loops, best of 3: 1.33 msec per loop
timeit("Z.view(np.int64)[...] = 0", globals())
# 100 loops, best of 3: 874 usec per loop
timeit("Z.view(np.float64)[...] = 0", globals())
# 100 loops, best of 3: 865 usec per loop
timeit("Z.view(np.complex128)[...] = 0", globals())
# 100 loops, best of 3: 841 usec per loop
timeit("Z.view(np.int8)[...] = 0", globals())
# 100 loops, best of 3: 630 usec per loop
```

有趣的是，清除所有值的显而易见的方法并不是最快的。通过将数组转换为更大的数据类型（如 `np.float64`），我们获得了 25% 的速度提升。但是，通过将数组视为字节数组 (`np.int8`)，我们获得了 50% 的提升。这种加速的原因在于内部 numpy 机制和编译器优化。这个简单的例子说明了 numpy 的哲学，我们将在下面的下一节中看到。

## 内存布局 (Memory layout)

Numpy 文档非常清楚地定义了 `ndarray` 类：

> `ndarray` 类的实例由计算机内存的一个连续一维段（由数组拥有，或由某个其他对象拥有）以及将 N 个整数映射到块中项目位置的索引方案组成。

换句话说，数组主要是一个连续的内存块，其部分可以使用索引方案进行访问。这种索引方案反过来由形状和数据类型定义，这正是您定义新数组时所需要的：

```python
Z = np.arange(9).reshape(3,3).astype(np.int16)
```

在这里，我们知道 Z 的 itemsize 是 2 字节 (int16)，形状是 (3,3)，维度数是 2 (`len(Z.shape)`).

```python
print(Z.itemsize)
# 2
print(Z.shape)
# (3, 3)
print(Z.ndim)
# 2
```

此外，由于 Z 不是视图，我们可以推导出数组的步幅 (strides)，它定义了在遍历数组时每个维度中步进的字节数。

```python
strides = Z.shape[1]*Z.itemsize, Z.itemsize
print(strides)
# (6, 2)
print(Z.strides)
# (6, 2)
```

有了所有这些信息，我们知道如何访问特定项目（由索引元组设计），更准确地说，如何计算开始和结束偏移量：

```python
offset_start = 0
for i in range(ndim):
    offset_start += strides[i]*index[i]
offset_end = offset_start + Z.itemsize
```

让我们看看使用 `tobytes` 转换方法是否正确：

```python
Z = np.arange(9).reshape(3,3).astype(np.int16)
index = 1,1
print(Z[index].tobytes())
# b'\x04\x00'
offset = 0
for i in range(Z.ndim):
    offset += Z.strides[i]*index[i]
offset_start = offset
offset_end = offset + Z.itemsize
print(Z.tobytes()[offset_start:offset_end])
# b'\x04\x00'
```

该数组实际上可以从不同的角度（即布局）来考虑。

**项目布局**
```
               shape[1]
                 (=3)
            ┌───────────┐

         ┌  ┌───┬───┬───┐  ┐
         │  │ 0 │ 1 │ 2 │  │
         │  ├───┼───┼───┤  │
shape[0] │  │ 3 │ 4 │ 5 │  │ len(Z)
 (=3)    │  ├───┼───┼───┤  │  (=3)
         │  │ 6 │ 7 │ 8 │  │
         └  └───┴───┴───┘  ┘
```

**扁平化项目布局**
```
┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┘

└───────────────────────────────────┘
               Z.size
                (=9)
```

**内存布局 (C 顺序, 大端)**
```
                         strides[1]
                           (=2)
                  ┌─────────────────────┐

          ┌       ┌──────────┬──────────┐ ┐
          │ p+00: │ 00000000 │ 00000000 │ │
          │       ├──────────┼──────────┤ │
          │ p+02: │ 00000000 │ 00000001 │ │ strides[0]
          │       ├──────────┼──────────┤ │   (=2x3)
          │ p+04  │ 00000000 │ 00000010 │ │
          │       ├──────────┼──────────┤ ┘
          │ p+06  │ 00000000 │ 00000011 │
          │       ├──────────┼──────────┤
Z.nbytes  │ p+08: │ 00000000 │ 00000100 │
(=3x3x2)  │       ├──────────┼──────────┤
          │ p+10: │ 00000000 │ 00000101 │
          │       ├──────────┼──────────┤
          │ p+12: │ 00000000 │ 00000110 │
          │       ├──────────┼──────────┤
          │ p+14: │ 00000000 │ 00000111 │
          │       ├──────────┼──────────┤
          │ p+16: │ 00000000 │ 00001000 │
          └       └──────────┴──────────┘

                  └─────────────────────┘
                        Z.itemsize
                     Z.dtype.itemsize
                           (=2)
```

如果我们现在取 Z 的切片，结果是基本数组 Z 的视图：
```python
V = Z[::2,::2]
```

这种视图是使用形状、dtype 和步幅指定的，因为步幅不能再仅从 dtype 和形状推导出来。

## 视图和副本 (Views and copies)

视图和副本是优化数值计算的重要概念。即使我们在上一节中已经操作过它们，整个故事也要复杂一些。

### 直接和间接访问 (Direct and indirect access)

首先，我们必须区分索引和花式索引 (fancy indexing)。第一种总是返回视图，而第二种总是返回副本。这种区别很重要，因为在第一种情况下，修改视图会修改基本数组，而在第二种情况下则不会：

```python
Z = np.zeros(9)
Z_view = Z[:3]
Z_view[...] = 1
print(Z)
# [ 1.  1.  1.  0.  0.  0.  0.  0.  0.]

Z = np.zeros(9)
Z_copy = Z[[0,1,2]]
Z_copy[...] = 1
print(Z)
# [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
```

因此，如果你需要花式索引，最好保留一份花式索引的副本（特别是如果计算它很复杂），并使用它：

```python
Z = np.zeros(9)
index = [0,1,2]
Z[index] = 1
print(Z)
# [ 1.  1.  1.  0.  0.  0.  0.  0.  0.]
```

如果您不确定索引的结果是视图还是副本，您可以检查结果的基 (base) 是什么。如果它是 None，则结果是副本：

```python
Z = np.random.uniform(0,1,(5,5))
Z1 = Z[:3,:]
Z2 = Z[[0,1,2], :]
print(np.allclose(Z1,Z2))
# True
print(Z1.base is Z)
# True
print(Z2.base is Z)
# False
print(Z2.base is None)
# True
```

请注意，一些 numpy 函数在可能的情况下返回视图（例如 `ravel`），而另一些则总是返回副本（例如 `flatten`）：

```python
Z = np.zeros((5,5))
print(Z.ravel().base is Z)
# True
print(Z[::2,::2].ravel().base is Z)
# False
print(Z.flatten().base is Z)
# False
```

### 临时副本 (Temporary copy)

可以像上一节那样显式创建副本，但最常见的情况是隐式创建中间副本。这是在对数组进行算术运算时的情况：

```python
X = np.ones(10, dtype=np.int)
Y = np.ones(10, dtype=np.int)
A = 2*X + 2*Y
```

在上面的例子中，创建了三个中间数组。一个用于保存 `2*X` 的结果，一个用于保存 `2*Y` 的结果，最后一个用于保存 `2*X+2*Y` 的结果。在这个特定的例子中，数组足够小，这并没有真正的区别。但是，如果您的数组很大，那么您必须小心这些表达式，并想知道是否可以用不同的方式来做。例如，如果只有最终结果很重要，并且之后不需要 X 或 Y，替代解决方案是：

```python
X = np.ones(10, dtype=np.int)
Y = np.ones(10, dtype=np.int)
np.multiply(X, 2, out=X)
np.multiply(Y, 2, out=Y)
np.add(X, Y, out=X)
```

使用此替代解决方案，不会创建临时数组。问题是还有许多其他情况需要创建此类副本，这会影响性能，如下例所示：

```python
X = np.ones(1000000000, dtype=np.int)
Y = np.ones(1000000000, dtype=np.int)
timeit("X = X + 2.0*Y", globals())
# 100 loops, best of 3: 3.61 ms per loop
timeit("X = X + 2*Y", globals())
# 100 loops, best of 3: 3.47 ms per loop
timeit("X += 2*Y", globals())
# 100 loops, best of 3: 2.79 ms per loop
timeit("np.add(X, Y, out=X); np.add(X, Y, out=X)", globals())
# 1000 loops, best of 3: 1.57 ms per loop
```

## 结论 (Conclusion)

作为结论，我们将做一个练习。给定两个向量 Z1 和 Z2。我们想知道 Z2 是否是 Z1 的视图，如果是，这个视图是什么？

```python
Z1 = np.arange(10)
Z2 = Z1[1:-1:2]
```

首先，我们需要检查 Z1 是否是 Z2 的基：

```python
print(Z2.base is Z1)
# True
```

此时，我们知道 Z2 是 Z1 的视图，这意味着 Z2 可以表示为 `Z1[start:stop:step]`。困难在于找到 start, stop 和 step。对于 step，我们可以使用任何数组的 strides 属性，该属性给出了在每个维度中从一个元素到另一个元素的字节数。在我们的例子中，因为两个数组都是一维的，我们可以直接比较第一个 stride：

```python
step = Z2.strides[0] // Z1.strides[0]
print(step)
# 2
```

接下来的困难是找到开始和结束索引。为此，我们可以利用 `byte_bounds` 方法，该方法返回指向数组端点的指针。

```python
offset_start = np.byte_bounds(Z2)[0] - np.byte_bounds(Z1)[0]
print(offset_start) # bytes
# 8

offset_stop = np.byte_bounds(Z2)[-1] - np.byte_bounds(Z1)[-1]
print(offset_stop) # bytes
# -16
```

将这些偏移量转换为索引很简单，只需使用 itemsize 并考虑到 offset_stop 是负数（Z2 数组的末端逻辑上小于 Z1 数组的末端）。因此，我们需要加上 Z1 的 items 大小以获得正确的结束索引。

```python
start = offset_start // Z1.itemsize
stop = Z1.size + offset_stop // Z1.itemsize
print(start, stop, step)
# 1, 8, 2
```

最后我们测试我们的结果：

```python
print(np.allclose(Z1[start:stop:step], Z2))
# True
```

作为一个练习，你可以通过考虑以下因素来改进这个最初且非常简单的实现：
* 负步长
* 多维数组

[练习的解决方案](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/find_index.py)

---
[返回目录](README.md)
