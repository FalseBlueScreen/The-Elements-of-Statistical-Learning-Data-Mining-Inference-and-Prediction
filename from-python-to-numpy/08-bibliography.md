# 参考文献 (Bibliography)

这是一个精选的 numpy 相关资源列表（文章、书籍和教程），涉及 numpy 的不同方面。有些非常特定于 numpy/Scipy，而另一些则提供了关于数值计算的更广泛视角。

## 教程 (Tutorials)

* [100 Numpy exercises](http://www.labri.fr/perso/nrougier/teaching/numpy.100/index.html), Nicolas P. Rougier, 2016.
* [Numpy tutorial](http://www.labri.fr/perso/nrougier/teaching/numpy/numpy.html), Nicolas P. Rougier, 2015.
* [Python course](http://www.python-course.eu/numpy.php), Bernd Klein, 2015.
* [An introduction to Numpy and Scipy](https://engineering.ucsb.edu/~shell/che210d/numpy.pdf), M. Scott Shell, 2014.
* [Python Numpy tutorial](http://cs231n.github.io/python-numpy-tutorial/), Justin Johnson, 2014.
* [Quickstart tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html), Numpy developers, 2009.
* [Numpy medkits](http://mentat.za.net/numpy/numpy_advanced_slides/), Stéfan van der Walt, 2008.

## 文章 (Articles)

* **The NumPy array: a structure for efficient numerical computation**, Stéfan van der Walt, Chris Colbert & Gael Varoquaux, Computing in Science and Engineering, 13(2), 2011. [PDF](https://hal.inria.fr/inria-00564007/document)
  > 在 Python 世界中，NumPy 数组是数值数据的标准表示，并能够使用高级语言高效实现数值计算。正如这项工作所表明的那样，可以通过三种技术提高 NumPy 性能：向量化计算、避免在内存中复制数据以及最小化操作计数。

* **Vectorised algorithms for spiking neural network simulation**, Romain Brette & Dan F. M. Goodman, Neural Computation, 23(6), 2010. [PDF](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.397.6097)
  > 高级语言 (Matlab, Python) 在神经科学中很流行，因为它们灵活且能加速开发。然而，对于模拟脉冲神经网络，解释的成本是一个瓶颈。我们描述了一组算法，使用基于向量的操作高效地模拟具有高级语言的大型脉冲神经网络。这些算法构成了 Brian 的核心，Brian 是用 Python 语言编写的脉冲神经网络模拟器。向量化模拟使得将高级语言的灵活性与通常与编译语言相关的计算效率相结合成为可能。

* **Python for Scientific Computing**, Travis E. Oliphant, Computing in Science & Engineering, 9(3), 2007. [PDF](http://dl.acm.org/citation.cfm?id=1251830)
  > 本身，Python 是用于其他语言编写的科学代码的出色 "转向" 语言。然而，通过额外的基本工具，Python 转变为适合科学和工程代码的高级语言，这些代码通常足够快以便立即有用，但也足够灵活，可以通过额外的扩展来加速。

## 书籍 (Books)

* **[SciPy Lecture Notes](http://www.scipy-lectures.org/)**, Gaël Varoquaux, Emmanuelle Gouillart, Olav Vahtras et al., 2016.
  > 一份学习用 Python 进行数值、科学和数据的文档。关于科学 Python 生态系统的教程：核心工具和技术的快速介绍。不同的章节对应于 1 到 2 小时的课程，专业水平不断提高，从初学者到专家。

* **[Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do)**, Jake van der Plas, O'Reilly, 2016.
  > Python 数据科学手册提供了对数据密集型科学、研究和发现的核心计算和统计方法的广度参考。想要有效地使用 Python 进行数据科学任务的具有编程背景的人将学习如何面对各种问题：例如，如何将这种数据格式读入脚本？如何操作、转换和清理这些数据？如何使用这些数据来获得见解、回答问题或构建统计或机器学习模型？

* **[Elegant SciPy: The Art of Scientific Python](http://shop.oreilly.com/product/0636920038481.do)**, Juan Nunez-Iglesias, Stéfan van der Walt, Harriet Dashnow, O'Reilly, 2016.
  > 欢迎来到科学 Python 及其社区！通过这本实用的书，您将学习 SciPy 和相关库的基本部分，并体验可在实践中使用的优美、易读的代码。越来越多的科学家正在编程，SciPy 库在这里提供帮助。找到有用的函数并正确、高效地使用它们，以及使用易于阅读的代码是两件非常不同的事情。您将通过一些最好的可用代码示例来学习，这些代码选自涵盖广泛的 SciPy 和相关库——包括 scikit-learn, scikit-image, toolz, 和 pandas。

* **[Learning IPython for Interactive Computing and Data Visualization](https://www.packtpub.com/big-data-and-business-intelligence/learning-ipython-interactive-computing-and-data-visualization-sec)**, Cyrille Rossant, Packt Publishing, 2015.
  > 本书是 Python 数据分析平台的初学者友好指南。在介绍了 Python 语言、IPython 和 Jupyter Notebook 之后，您将学习如何分析和可视化现实世界示例中的数据，如何为 Notebook 中的图像处理创建图形用户界面，以及如何使用 NumPy, Numba, Cython, 和 ipyparallel 进行科学模拟的快速数值计算。在本书结束时，您将能够对各种数据进行深入分析。

* **[SciPy and NumPy](https://www.safaribooksonline.com/library/view/scipy-and-numpy/9781449361600/)**, Eli Bressert, O'Reilly Media, Inc., 2012.
  > 您是 SciPy 和 NumPy 的新手吗？您想通过示例和简明扼要的介绍快速轻松地学习它吗？那么这本书适合你。您将通过在线文档的复杂性，发现如何轻松掌握这些 Python 库。

* **[Python for Data Analysis](http://shop.oreilly.com/product/0636920023784.do)**, Wes McKinney, O'Reilly Media, Inc., 2012.
  > 寻找在 Python 中操作、处理、清理和处理结构化数据的完整说明？这本实践书籍充满了实际案例研究，向您展示了如何使用多个 Python 库有效地解决广泛的数据分析问题。

* **[Guide to NumPy](http://csc.ucdavis.edu/~chaos/courses/nlp/Software/NumPyBook.pdf)**, Travis Oliphant, 2006.
  > 本书仅简要概述了围绕 NumPy 中基本对象的一些基础设施，以提供旧 Numeric 包中包含的附加功能（即 LinearAlgebra, RandomArray, FFT）。NumPy 中的这个基础设施包括基本的线性代数例程、傅立叶变换功能和随机数生成器。此外，f2py 模块在其自己的文档中进行了描述，因此在本书的第二部分中仅简要提及。

---
[返回目录](README.md)
