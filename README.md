# ML-demo

## 线性回归示例

`linear_regression_demo.py` 演示了如何只用 NumPy 就完成一个最简单的线性回归任务：

- 生成一批可重复的合成数据（单特征）。
- 使用正规方程（Normal Equation）推导出解析解，得到截距和斜率。
- 在训练集与测试集上计算均方误差（MSE）和决定系数（R^2）。
- 可以对自定义输入值进行预测。

### 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install numpy
```

### 运行

```bash
python linear_regression_demo.py --samples 100 --noise 3 --predict 2 5 8
```

输出示例：

```
=== Linear Regression Demo ===
Learned model: y = 11.832 + 3.537 * x
Train MSE: 8.771 | Train R^2: 0.959
 Test MSE: 8.062 |  Test R^2: 0.961

Predictions for custom inputs:
x =   2.00 -> y_hat = 18.906
x =   5.00 -> y_hat = 29.519
x =   8.00 -> y_hat = 40.130
```

可以通过 `--samples`、`--noise`、`--seed` 参数控制数据的规模、噪声大小以及随机性，方便快速做不同实验。 

## 无监督学习示例

`kmeans_demo.py` 展示了如何用 K-Means 聚类对合成数据进行无监督学习实验：

- 随机生成多个高斯簇（自定义样本数、簇数、特征维度与标准差）。
- 完整实现 Lloyd 算法（随机初始化、多次迭代、容差收敛、空簇重置）。
- 打印收敛迭代次数、簇中心、簇大小以及基于真实标签的粗略聚类准确率。

### 运行

```bash
python kmeans_demo.py --samples 400 --clusters 5 --features 2 --cluster-std 1.0
```

可以通过 `--max-iter` 与 `--tol` 控制收敛条件，用 `--seed` 确保实验可重复。 

## 深度学习示例

`deep_learning_demo.py` 使用 PyTorch 构建一个两层全连接网络，在随机生成的多簇数据上完成多分类任务：

- `generate_clusters` 动态合成带噪声的二维数据集，可配置样本数、类别数与噪声强度。
- `SimpleClassifier` 包含若干隐藏层（默认 64-64），使用 ReLU 激活，最后接 `CrossEntropyLoss`。
- 训练循环展示损失、训练/测试准确率，并输出若干测试样本的预测概率。

### 依赖安装

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121  # 根据本地环境选择合适的指令
```

### 运行

```bash
python deep_learning_demo.py --samples 600 --classes 3 --epochs 200 --lr 0.005
```

可通过 `--hidden 128 64` 自定义网络结构，`--device cuda` 指定显卡训练，`--test-ratio` 控制训练/测试划分比例。 
