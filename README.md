weak-supervision
=================

**Object-detection under weak supervision**

每种训练流程暂时还是使用代码表示。如果要创新训练流程, 需要写新的Trainer.

提ROI提议(proposals)、提取特征(feature)、分类器(detector/classifer) 已经抽象出接口(interface)。

运行
------------
先需要安装:

```bash
python setup.py develop --user
```

然后可以运行(可能需要将`${HOME}/.local/bin`加入环境变量`${PATH}`):

```bash
wks-train <配置文件的路径>
```
