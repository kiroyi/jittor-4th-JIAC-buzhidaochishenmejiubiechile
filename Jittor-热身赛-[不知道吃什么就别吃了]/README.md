# Jittor 记图挑战热身赛 
jittor-不知道吃什么就别吃了-记图挑战热身赛

![image](result.png)

## 简介
本项目包含记图挑战热身赛的代码。
项目特点：将在数字图片数据集 MNIST 上训练 Conditional GAN（Conditional generative adversarial nets）模型，通过输入一个随机向量 z 和额外的辅助信息 y (如类别标签)，生成特定数字的图像。

## 安装 
| 介绍基本的硬件需求、运行环境、依赖安装方法
#### 运行环境
-操作系统: Ubuntu >= 16.04 或 Windows Subsystem of Linux（WSL）

-Python：版本 >= 3.7

-C++编译器 ：g++ （>=5.4.0），clang （>=8.0）

-GPU 编译器（可选）：nvcc >=10.0

-GPU 加速库（可选）：cudnn-dev 

#### 安装依赖
执行以下命令安装 python 依赖

sudo apt install python3.7-dev libomp-dev

python3.7 -m pip install jittor

python3.7 -m jittor.test.test_example


## 训练
| 训练直接运行CGAN即可。number = '382268381621'，number可改为想要生成的特定数字。
