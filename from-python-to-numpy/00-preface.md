# 前言 (Preface)

## 关于作者 (About the author)

Nicolas P. Rougier 是法国国家计算机科学与自动化研究所 (Inria) 的全职研究科学家。Inria 是法国负责计算机科学和控制研究的国家机构。这是一个在研究与高等教育部以及经济、财政和工业部双重监督下的公共科学技术机构 (EPST)。Nicolas P. Rougier 在 Mnemosyne 项目组工作，该项目组位于综合神经科学和计算神经科学的前沿，与神经退行性疾病研究所、波尔多计算机科学研究实验室 (LaBRI)、波尔多大学和国家科学研究中心 (CNRS) 合作。

他使用 Python 已超过 15 年，使用 Numpy 进行神经科学建模、机器学习和高级可视化 (OpenGL) 已超过 10 年。Nicolas P. Rougier 是多个在线资源和教程 (Matplotlib, Numpy, OpenGL) 的作者，并在波尔多大学以及世界各地的各种会议和学校 (SciPy, EuroScipy 等) 教授 Python, Numpy 和科学可视化。他还是热门文章 "Ten Simple Rules for Better Figures" 和热门 Matplotlib 教程的作者。

## 关于本书 (About this book)

本书使用 reStructuredText 格式编写，并使用 Python 的 docutils 包中的 rst2html.py 命令行工具生成。

如果您想重建 html 输出，请从顶层目录输入：
```bash
$ rst2html.py --link-stylesheet --cloak-email-addresses \
              --toc-top-backlinks --stylesheet=book.css \
              --stylesheet-dirs=. book.rst book.html
```

源代码可从 [https://github.com/rougier/from-python-to-numpy](https://github.com/rougier/from-python-to-numpy) 获取。

## 先决条件 (Prerequisites)

这不是一本 Python 初学者指南，您应该具备中级 Python 水平，并且最好具备 Numpy 的初级水平。如果情况并非如此，请查看参考文献中列出的精选资源列表。

## 约定 (Conventions)

我们将使用通常的命名约定。如果没有明确说明，每个脚本都应按如下方式导入 numpy, scipy 和 matplotlib：
```python
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
```

我们将使用不同软件包的最新版本（在撰写本文时，即 2017 年 1 月）：
```
Packages  Version
Python     3.6.0
Numpy      1.12.0
Scipy      0.18.1
Matplotlib 2.0.0
```

## 如何贡献 (How to contribute)

如果您想为本书做出贡献，您可以：
* 审查章节（请联系我）
* 报告问题 ([https://github.com/rougier/from-python-to-numpy/issues](https://github.com/rougier/from-python-to-numpy/issues))
* 建议改进 ([https://github.com/rougier/from-python-to-numpy/pulls](https://github.com/rougier/from-python-to-numpy/pulls))
* 纠正英语 ([https://github.com/rougier/from-python-to-numpy/issues](https://github.com/rougier/from-python-to-numpy/issues))
* 为本书设计更好、响应更快的 html 模板。
* 为项目加星标 ([https://github.com/rougier/from-python-to-numpy](https://github.com/rougier/from-python-to-numpy))

## 出版 (Publishing)

如果您是有意出版这本书的编辑，如果您同意将此版本及所有后续版本保持开放获取（即在线地址不变），知道如何处理 reStructuredText（Word 不是一种选择），您能提供真正的增值服务以及支持服务，更重要的是，您拥有真正令人惊叹的 latex 书籍模板（请注意，我对排版和设计有点挑剔：Edward Tufte 是我的英雄），请联系我。还在吗？

## 许可证 (License)

**书籍**

本作品根据 [Creative Commons Attribution-Non Commercial-Share Alike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) 进行许可。您可以自由地：
* 分享 — 在任何媒介或格式下复制和分发材料
* 改编 — 重新混合、转换和基于该材料进行构建

只要您遵守许可条款，许可人就不能撤销这些自由。

**代码**

代码根据 OSI 批准的 BSD 2-Clause License 许可。

---
[返回目录](README.md)
