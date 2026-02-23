# 自定义向量化 (Custom vectorization)

## 简介 (Introduction)

Numpy 的优势之一是它可以用来构建新对象或子类化 `ndarray` 对象。后一个过程有点乏味，但值得努力，因为它允许您改进 `ndarray` 对象以适应您的问题。我们将在下一节中研究两个实际案例（类型列表和内存感知数组），它们在 [glumpy](http://glumpy.github.io/) 项目（由我维护）中被广泛使用，而最后一个案例（双精度数组）是一个更学术的案例。

## 类型列表 (Typed list)

类型列表（也称为不规则数组）是具有相同数据类型（在 numpy 意义上）的项列表。它们提供列表和 `ndarray` API（当然有一些限制），但由于它们各自的 API 在某些情况下可能不兼容，我们必须做出选择。例如，关于 `+` 运算符，我们将选择使用 numpy API，即将值添加到每个单独的项，而不是通过追加新项来扩展列表 (1)。

```python
>>> l = TypedList([[1,2], [3]])
>>> print(l)
[1, 2], [3]
>>> print(l+1)
[2, 3], [4]
```

从列表 API 中，我们希望我们的新对象提供无缝插入、追加和删除项目的可能性。

### 创建 (Creation)

由于对象本质上是动态的，因此提供一种足够强大的通用创建方法以避免以后的操作非常重要。这种操作，例如插入/删除，会花费大量操作，我们要避免它们。这是创建 `TypedList` 对象的提案（众多提案之一）。

```python
def __init__(self, data=None, sizes=None, dtype=float)
    """
    Parameters
    ----------

    data : array_like
        An array, any object exposing the array interface, an object
        whose __array__ method returns an array, or any (nested) sequence.

    sizes:  int or 1-D array
        If `itemsize is an integer, N, the array will be divided
        into elements of size N. If such partition is not possible,
        an error is raised.

        If `itemsize` is 1-D array, the array will be divided into
        elements whose successive sizes will be picked from itemsize.
        If the sum of itemsize values is different from array size,
        an error is raised.

    dtype: np.dtype
        Any object that can be interpreted as a numpy data type.
    """
```

此 API 允许创建一个空列表或从一些外部数据创建一个列表。请注意，在后一种情况下，我们需要指定如何将数据划分为多个项，否则它们将分裂为 1 大小的项。它可以是常规分区（即每个项长 2 个数据）或自定义分区（即数据必须拆分为大小为 1、2、3 和 4 的项）。

```python
>>> L = TypedList([[0], [1,2], [3,4,5], [6,7,8,9]])
>>> print(L)
[ [0] [1 2] [3 4 5] [6 7 8] ]

>>> L = TypedList(np.arange(10), [1,2,3,4])
[ [0] [1 2] [3 4 5] [6 7 8] ]
```

此时，问题是是否要子类化 `ndarray` 类或使用内部 `ndarray` 来存储我们的数据。在我们的具体情况下，子类化 `ndarray` 并没有真正的意义，因为我们并不真正想提供 `ndarray` 接口。相反，我们将使用 `ndarray` 来存储列表数据，这种设计选择将为我们提供更大的灵活性。

```
╌╌╌╌┬───┐┌───┬───┐┌───┬───┬───┐┌───┬───┬───┬───┬╌╌╌╌╌
    │ 0 ││ 1 │ 2 ││ 3 │ 4 │ 5 ││ 6 │ 7 │ 8 │ 9 │
 ╌╌╌┴───┘└───┴───┘└───┴───┴───┘└───┴───┴───┴───┴╌╌╌╌╌╌
   item 1  item 2    item 3         item 4
```

为了存储每个项的限制，我们将使用一个 `items` 数组，它将负责存储每个项的位置（开始和结束）。对于列表的创建，有两种不同的情况：没有给出数据或给出了一些数据。第一种情况很容易，只需要创建 `_data` 和 `_items` 数组。请注意，它们的大小不为空，因为每次插入新项时调整数组大小太昂贵了。相反，最好保留一些空间。

**第一种情况**。没有给出数据，只有 dtype。

```python
self._data = np.zeros(512, dtype=dtype)
self._items = np.zeros((64,2), dtype=int)
self._size = 0
self._count = 0
```

**第二种情况**。给出了一些数据以及项目大小列表（有关其他情况，请参阅下面的完整代码）

```python
self._data = np.array(data, copy=False)
self._size = data.size
self._count = len(sizes)
indices = sizes.cumsum()
self._items = np.zeros((len(sizes),2),int)
self._items[1:,0] += indices[:-1]
self._items[0:,1] += indices
```

### 访问 (Access)

一旦完成，每个列表方法只需要一点计算，并在获取、插入或设置项时使用不同的键。这是 `__getitem__` 方法的代码。没有真正的困难，除了可能的负步长：

```python
def __getitem__(self, key):
    if type(key) is int:
        if key < 0:
            key += len(self)
        if key < 0 or key >= len(self):
            raise IndexError("Tuple index out of range")
        dstart = self._items[key][0]
        dstop  = self._items[key][1]
        return self._data[dstart:dstop]

    elif type(key) is slice:
        istart, istop, step = key.indices(len(self))
        if istart > istop:
            istart,istop = istop,istart
        dstart = self._items[istart][0]
        if istart == istop:
            dstop = dstart
        else:
            dstop  = self._items[istop-1][1]
        return self._data[dstart:dstop]

    elif isinstance(key,str):
        return self._data[key][:self._size]

    elif key is Ellipsis:
        return self.data

    else:
        raise TypeError("List indices must be integers")
```

### 练习 (Exercise)

列表的修改有点复杂，因为它需要正确管理内存。由于它没有真正的困难，我们将其留作读者的练习。对于懒惰的人，您可以看看下面的代码。注意负步长、键范围和数组扩展。当需要扩展底层数组时，最好将其扩展得比必要的更多，以避免未来的扩展。

**setitem**
```
L = TypedList([[0,0], [1,1], [0,0]])
L[1] = 1,1,1

╌╌╌╌┬───┬───┐┌───┬───┐┌───┬───┬╌╌╌╌╌
    │ 0 │ 0 ││ 1 │ 1 ││ 2 │ 2 │
 ╌╌╌┴───┴───┘└───┴───┘└───┴───┴╌╌╌╌╌╌
     item 1   item 2   item 3

╌╌╌╌┬───┬───┐┌───┬───┲━━━┓┌───┬───┬╌╌╌╌╌
    │ 0 │ 0 ││ 1 │ 1 ┃ 1 ┃│ 2 │ 2 │
 ╌╌╌┴───┴───┘└───┴───┺━━━┛└───┴───┴╌╌╌╌╌╌
     item 1     item 2     item 3
```

**delitem**
```
L = TypedList([[0,0], [1,1], [0,0]])
del L[1]

╌╌╌╌┬───┬───┐┏━━━┳━━━┓┌───┬───┬╌╌╌╌╌
    │ 0 │ 0 │┃ 1 ┃ 1 ┃│ 2 │ 2 │
 ╌╌╌┴───┴───┘┗━━━┻━━━┛└───┴───┴╌╌╌╌╌╌
     item 1   item 2   item 3

╌╌╌╌┬───┬───┐┌───┬───┬╌╌╌╌╌
    │ 0 │ 0 ││ 2 │ 2 │
 ╌╌╌┴───┴───┘└───┴───┴╌╌╌╌╌╌
     item 1    item 2
```

**insert**
```
L = TypedList([[0,0], [1,1], [0,0]])
L.insert(1, [3,3])

╌╌╌╌┬───┬───┐┌───┬───┐┌───┬───┬╌╌╌╌╌
    │ 0 │ 0 ││ 1 │ 1 ││ 2 │ 2 │
 ╌╌╌┴───┴───┘└───┴───┘└───┴───┴╌╌╌╌╌╌
     item 1   item 2   item 3

╌╌╌╌┬───┬───┐┏━━━┳━━━┓┌───┬───┐┌───┬───┬╌╌╌╌╌
    │ 0 │ 0 │┃ 3 ┃ 3 ┃│ 1 │ 1 ││ 2 │ 2 │
 ╌╌╌┴───┴───┘┗━━━┻━━━┛└───┴───┘└───┴───┴╌╌╌╌╌╌
     item 1   item 2   item 3   item 4
```

### 来源 (Sources)

* [array_list.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/array_list.py) (练习的解决方案)

## 内存感知数组 (Memory aware array)

### Glumpy

[Glumpy](http://glumpy.github.io/) 是一个基于 Python 的 OpenGL 交互式可视化库，其目标是使创建快速、可扩展、美观、交互式和动态的可视化变得容易。

![](https://www.labri.fr/perso/nrougier/from-python-to-numpy/data/galaxy.png)
![](https://www.labri.fr/perso/nrougier/from-python-to-numpy/data/tiger.png)

Glumpy 基于与 numpy 数组的紧密无缝集成。这意味着您可以像使用常规 numpy 数组一样操作 GPU 数据，glumpy 将负责其余的工作。但一个例子胜过千言万语：

```python
from glumpy import gloo

dtype = [("position", np.float32, 2),  # x,y
         ("color",    np.float32, 3)]  # r,g,b
V = np.zeros((3,3),dtype).view(gloo.VertexBuffer)
V["position"][0,0] = 0.0, 0.0
V["position"][1,1] = 0.0, 0.0
```

V 是一个 `VertexBuffer`，它既是 `GPUData` 又是 `numpy` 数组。当 V 被修改时，glumpy 负责计算自上次上传到 GPU 内存以来的最小连续脏内存块。当此缓冲区要在 GPU 上使用时，glumpy 会在最后一刻负责上传“脏”区域。这意味着如果你从不使用 V，什么也不会上传到 GPU！在上面的情况下，最后计算的“脏”区域由从偏移量 0 开始的 88 个字节组成，如下图所示：

![](https://www.labri.fr/perso/nrougier/from-python-to-numpy/data/GPUData.png)

> **注意**
>
> 当创建一个缓冲区时，它被标记为完全脏，但为了说明，假设这里不是这种情况。

因此，Glumpy 最终将上传 88 个字节，而实际上只修改了 16 个字节。您可能想知道这是否最佳。实际上，大多数时候它是，因为将一些数据上传到缓冲区需要在 GL 端进行大量操作，并且每个调用都有固定成本。

### 数组子类化 (Array subclass)

正如在[子类化 ndarray 文档](https://docs.scipy.org/doc/numpy/user/basics.subclassing.html)中所解释的，子类化 `ndarray` 变得复杂，因为 `ndarray` 类的新实例可以通过三种不同的方式产生：
* 显式构造函数调用
* 视图转换
* 从模板新建

然而，我们的情况更简单，因为我们只对视图转换感兴趣。因此，我们只需要定义 `__new__` 方法，该方法将在每次实例创建时被调用。因此，`GPUData` 类将配备两个属性：
* `extents`：这表示视图相对于基础数组的完整范围。它存储为字节偏移量和字节大小。
* `pending_data`：这表示相对于 `extents` 属性的连续脏区域为（字节偏移量，字节大小）。

```python
class GPUData(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.ndarray.__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        pass

    def __array_finalize__(self, obj):
        if not isinstance(obj, GPUData):
            self._extents = 0, self.size*self.itemsize
            self.__class__.__init__(self)
            self._pending_data = self._extents
        else:
            self._extents = obj._extents
```

### 计算范围 (Computing extents)

每次请求数组的部分视图时，我们都需要在我们可以访问基础数组时计算此部分视图的范围。

```python
def __getitem__(self, key):
    Z = np.ndarray.__getitem__(self, key)
    if not hasattr(Z,'shape') or Z.shape == ():
        return Z
    Z._extents = self._compute_extents(Z)
    return Z

def _compute_extents(self, Z):
    if self.base is not None:
        base = self.base.__array_interface__['data'][0]
        view = Z.__array_interface__['data'][0]
        offset = view - base
        shape = np.array(Z.shape) - 1
        strides = np.array(Z.strides)
        size = (shape*strides).sum() + Z.itemsize
        return offset, offset+size
    else:
        return 0, self.size*self.itemsize
```

### 跟踪待处理数据 (Keeping track of pending data)

一个额外的困难是，我们不希望所有的视图都跟踪脏区域，而只希望基础数组跟踪。这就是为什么我们在 `__array_finalize__` 方法的第二种情况下不实例化 `self._pending_data` 的原因。这将在我们需要更新一些数据时处理，例如在 `__setitem__` 调用期间：

```python
def __setitem__(self, key, value):
    Z = np.ndarray.__getitem__(self, key)
    if Z.shape == ():
        key = np.mod(np.array(key)+self.shape, self.shape)
        offset = self._extents[0]+(key * self.strides).sum()
        size = Z.itemsize
        self._add_pending_data(offset, offset+size)
        key = tuple(key)
    else:
        Z._extents = self._compute_extents(Z)
        self._add_pending_data(Z._extents[0], Z._extents[1])
    np.ndarray.__setitem__(self, key, value)

def _add_pending_data(self, start, stop):
    base = self.base
    if isinstance(base, GPUData):
        base._add_pending_data(start, stop)
    else:
        if self._pending_data is None:
            self._pending_data = start, stop
        else:
            start = min(self._pending_data[0], start)
            stop = max(self._pending_data[1], stop)
            self._pending_data = start, stop
```

### 来源 (Sources)

* [gpudata.py](https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/gpudata.py)

## 结论 (Conclusion)

正如 numpy 网站上所解释的那样，numpy 是 Python 科学计算的基础包。然而，正如本章所说明的那样，numpy 的优势远不止于通用数据的多维容器。在一个案例中使用 `ndarray` 作为私有属性（`TypedList`）或在另一个案例中直接子类化 `ndarray` 类（`GPUData`）来跟踪内存，我们已经看到了如何扩展 numpy 的功能以适应非常具体的需求。限制只在于你的想象力和你的经验。

---
[返回目录](README.md)
