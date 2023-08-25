# qta_prj
2023 QTA Written Exam - Programming Task B

## 工程环境

使用学校集群训练。
1. GPU 为 NVIDIA A100 PCIe 40GB；
2. 操作系统为 Linux 64 位发行版 Ubuntu 18.04 LTS；
3. 处理器为 Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz。

## 项目架构

项目文件由以下几个部分组成：
1. qta_cifar.ipynb	主文件
2. model.py		    构建模型
3. trainer.py		训练模型
4. tester.py		单独测试模型
5. progbar.py	    可复用的训练进度条组件

## 模型使用

选型上，综合考虑数据集的复杂程度、各类模型的参数量等因素，使用基于 ResNet 的 [DenseNet](https://arxiv.org/abs/1608.06993) 作为训练模型。
DenseNet 提出了一个更激进的密集连接机制：即互相连接所有的层，具体来说就是每个层都会接受其前面所有层作为其额外的输入。
综合来看，DenseNet 的优势主要体现在以下几个方面：
1. 密集连接的方式使得 DenseNet 更容易训练。
2. DenseNet实现了特征重用，参数更小且计算更高效。
3. 由于特征复用，DenseNet 最后的分类器使用了低级特征。
在模型的参数及训练方式上，参照 DenseNet 原始论文的方法进行操作。

可一键运行 `qta_cifar.ipynb`，提供训练过程的可视化以及测试结果的图表绘制。