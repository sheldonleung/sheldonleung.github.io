---
title: Neural Networks and Deep Learning 第一章
date: 2019-08-30 10:28:04
categories: 读书笔记
tags: 神经网络
mathjax: true
---

记录今天的读书笔记，《Neural Networks and Deep Learning》第一章

<!--more-->

## 1.1 感知器

####  1.1.1 什么是感知器？

&nbsp;&nbsp;&nbsp;&nbsp;感知器是一种人工神经元，在20世纪五、六十年代由科学家Frank Rosenblatt发明。

&nbsp;&nbsp;&nbsp;&nbsp;一个感知器有一个或多个二进制输入($x_1,x_2,x_3$)，这些输入都有相应的**权重**($w_1,w_2,w_3$)，权重表示相应的输入对于输出有重要性的实数。感知器有一个二进制输出(0或者1)，感知器的输出则由分配权重后的信号量总和$\sum_jw_jx_j$⼩于或者⼤于**阈值**($threshold$)来决定。权重和阈值都是神经元的参数并且都是实数。

&nbsp;&nbsp;&nbsp;&nbsp;神经元的代数形式：

$$
output=\begin{cases}
0 && if \sum_jw_jx_j \leq threshold \\\\
1 && if \sum_jw_jx_j \gt threshold
\end{cases}
$$

&nbsp;&nbsp;&nbsp;&nbsp;感知器的代数形式可以进一步简化，将信号量总和$\sum_j w_jx_j$化成点积$w \cdot x$，阈值($threshold$)化成偏置$b=-threshold$：

$$
output= \begin{cases}
0 && if \quad w \cdot x + b \leq 0 \\\\
1 && if \quad w \cdot x + b > 0
\end{cases}
$$

&nbsp;&nbsp;&nbsp;&nbsp;在感知器中，任何权重或者偏置发生微小改变，都会引起输出发生巨大的改变。

![感知器示意图](https://take-care.oss-cn-shenzhen.aliyuncs.com/blog-img/Neural-Networks-and-Deep-Learning/img1.1-1.png)

<p align="center">感知器示意图</p>

## 1.2 S型神经元

### 1.2.1 S型神经元的工作原理

- 输入可以取0或者1之间的任意值
- 输出为$\sigma(w \cdot x + b)$，其中$\sigma(z) \equiv \frac{1}{1+e^{-z}}$，公式可写成$\frac{1}{1+exp(-\sum_jw_jx_j-b)}$

&nbsp;&nbsp;&nbsp;&nbsp;S型神经元是改进自感知器，在感知器的基础上引入偏导数，权重或者偏置做出的微小改动只引起输出的微小改动而不会引起输出的巨大改变。

$$
\Delta output \approx \sum_j \frac{\partial output}{\partial w_j}\Delta w_j + \frac{\partial output}{\partial b}\Delta b
$$

## 1.3 神经网络的架构

&nbsp;&nbsp;&nbsp;&nbsp;一个神经网络的结构主要由输入层，隐藏层和输出层构成。

![](https://take-care.oss-cn-shenzhen.aliyuncs.com/blog-img/Neural-Networks-and-Deep-Learning/img1.3-1.png)

&nbsp;&nbsp;&nbsp;&nbsp;像这种多层网络结构的被称为多层感知器，但是构成网络的神经元不是感知器而是S型神经元。

&nbsp;&nbsp;&nbsp;&nbsp;书中提到两种神经网络架构：

1. 前馈神经网络：将上一层的输出作为下一层的输入。上图就是一种前馈神经网络。

2. 递归神经网络：不同于前馈神经网络的是，递归神经网络的结构中存在反馈环路。

## 1.4 一个简单的分类手写数字的网络

&nbsp;&nbsp;&nbsp;&nbsp;首先将图像![](https://take-care.oss-cn-shenzhen.aliyuncs.com/blog-img/Neural-Networks-and-Deep-Learning/img1.4-1.png)分割成单独的$28 \times 28$的图像![](https://take-care.oss-cn-shenzhen.aliyuncs.com/blog-img/Neural-Networks-and-Deep-Learning/img1.4-2.png)

&nbsp;&nbsp;&nbsp;&nbsp;接下来使用一个三层神经网络来识别单个数字：

![神经网络的结构](https://take-care.oss-cn-shenzhen.aliyuncs.com/blog-img/Neural-Networks-and-Deep-Learning/img1.4-3.png)

<p align="center">网络的结构</p>
&nbsp;&nbsp;&nbsp;&nbsp;网络的输入层包含给输入像素的值进行编码的784个神经元；网络的第二层是一个隐藏层(示例中只设置了15个神经元)；输出层包含10个神经元。

&nbsp;&nbsp;&nbsp;&nbsp;为什么输出层为什么使用10个神经元而不适用4个神经元？

&nbsp;&nbsp;&nbsp;&nbsp;4个神经元每一个输出作为二进制的话，结果取决于它的输出更靠近0还是1，而且使用4个神经元一共有$2^4=16$种可能的输出。所以基于经验主义，使用10个神经元作为输出，识别效果更好

## 1.5 使用梯度下降算法进行学习

&nbsp;&nbsp;&nbsp;&nbsp;定义代价(损失)函数：

$$
C(w,b) \equiv \frac{1}{2n}\sum_x||y(x)-a||^2
$$


&nbsp;&nbsp;&nbsp;&nbsp;其中，$w$表示所有的网络中权重的集合，$b$是所有的偏置，$n$是训练输入数据的个数，$a$是表示当输入$x$时输出的向量，符号$||v||$是指向量$v$的模，$C$称为**二次代价函数**。

- 目标：找出尽可能小的权重和偏置，使$C(w,b) \approx 0$。

### 1.5.1 双自变量

![](https://take-care.oss-cn-shenzhen.aliyuncs.com/blog-img/Neural-Networks-and-Deep-Learning/img1.5.1-1.png)

&nbsp;&nbsp;&nbsp;&nbsp;想象有一个球在$v_1,v_2$方向分别移动很小的量，即$\Delta v_1$和$\large \Delta v_2$，球体的位置将会发生变化，通过微积分得到$C$的变化:

$$
\Delta C \approx \frac{\partial C}{\partial v_1}\Delta v_1 + \frac{\partial C}{\partial v_2}\Delta v_2
$$


&nbsp;&nbsp;&nbsp;&nbsp;找出$\Delta v_1$和$\Delta v_2$使得$\Delta C$的值为负，定义一下两个式子：

$$
\Delta v \equiv (\Delta v_1,\Delta v_2)^T
\\\\
\nabla C \equiv (\frac{\partial C}{\partial v_1},\frac{\partial C}{\partial v_2})^T
$$

​	&nbsp;&nbsp;&nbsp;&nbsp;其中，$\Delta v$为$v$变化的量，$\nabla C$表示梯度向量，$T$ 是转置符号。将以上三式重写，得到：

$$
\Delta C \approx \nabla C \cdot \Delta v
$$


​	&nbsp;&nbsp;&nbsp;&nbsp;在这个式子中，$\nabla C$把$v$的变化关联为$C$的变化，同时让我们知道如何选取$\Delta v$才能让$\Delta C$为负。

​	&nbsp;&nbsp;&nbsp;&nbsp;假设我们选取$\Delta v = -\eta\nabla C$，其中$\eta$是一个很小的正数(称为**学习速率**)。所以我们可以得到：

$$
\Delta C \approx -\eta \nabla C \cdot \nabla C = -\eta ||\nabla C||^2
\\\\
\because ||\nabla C||^2 \geq 0
\\\\
\therefore \Delta C \leq 0
$$

&nbsp;&nbsp;&nbsp;&nbsp;所以定义$\Delta v = -\eta\nabla C$为球体在梯度下降算法下的"运动定律"。也就是说我们使用方程$\Delta v = -\eta\nabla C$ 计算$\Delta v$来移动球体的位置，反复这样操作，持续减小$C$，最终会获得$C$的全局最小值。

&nbsp;&nbsp;&nbsp;&nbsp;为了使$\Delta C \approx \nabla C \cdot \Delta v$达到很好的近似度，我们需要选择足够小的学习速率$\eta$，不然将会以$\eta \gt 0$结束；同时学习速率又不能过小，这样梯度下降算法就会运行得很慢。因此需要选择一个合适的学习速率能够达到很好的近似度，同时算法又不至于过慢。

### 1.5.2 多自变量

&nbsp;&nbsp;&nbsp;&nbsp;假设$C$是一个有$m$个自变量的多元函数，那么$C$中的自变量变化为$\Delta v=(\Delta v_1,\cdots,\Delta v_m)^T$，则$\Delta C$为$\Delta C \approx \nabla C \cdot \Delta v$，梯度向量$\nabla C$为：$\nabla C \equiv (\frac{\partial C}{\partial v_1},\cdots,\frac{\partial C}{\partial v_m})^T$。与双自变量的情况类似，$\Delta v$我们可以选取$\Delta v = -\eta\nabla C$。

&nbsp;&nbsp;&nbsp;&nbsp;假设我们正努力去改变$\Delta v$来让$\large C$尽可能地减小，这相当于最小化$\Delta C \approx \nabla C \cdot \Delta v$。首先限制步长为小的固定值，即$||\Delta v|| = \epsilon, \epsilon > 0$。当步长固定时，我们要找到使得$C$减小最大的下降方向。可以证明，使得$\nabla C \cdot \Delta v$取得最小值的$\Delta v$为$\Delta v = -\eta\nabla C$，这里$\eta = \epsilon / ||\nabla C||$是有步长限制$||\Delta v|| = \epsilon$所决定的。因此，梯度下降法可以被视为一种在$C$下降最快的方向上做最微小变化的方法。

### 1.5.3 如何在神经网络中使用梯度下降算法进行学习？

&nbsp;&nbsp;&nbsp;&nbsp;其思想就是利用梯度下降算法去寻找能够使得方程$C(w,b) \equiv \frac{1}{2n}\sum_x||y(x)-a||^2$取得最小值的权重$w_k$偏置$b_l$。具体过程根据 **1.5.1 双自变量**，将两个$v$分量替换成$w_k$和$b_l$，即可得到：

$$
\Delta w_k = -\eta \frac{\partial C}{\partial w_k}
\\\\
\Delta b_l = -\eta \frac{\partial C}{\partial b_l}
$$
&nbsp;&nbsp;&nbsp;&nbsp;剩下的思路都与1.5.1相似，这里就不在赘述。

### 1.5.4 随机梯度下降算法

&nbsp;&nbsp;&nbsp;&nbsp;其思想就是通过随机选取小量的$m$个训练输入样本$X_1,X_2,\cdots,X_m$，这些称为**小批量数据(mini-batch)**。假设样本数量$m$足够大，我们期望$\nabla C_{X_j}$的平均值大致相等于整个$\nabla C_x$的平均值：
$$
\frac{\sum_{j=1}^{m}\nabla C_{X_j}}{m} \approx \frac{\sum_x \nabla C_x}{n} = \nabla C
$$
&nbsp;&nbsp;&nbsp;&nbsp;化简得到
$$
\nabla C \approx \frac{1}{m}\sum_{j=1}^m\nabla C_{X_j}
$$
&nbsp;&nbsp;&nbsp;&nbsp;将随机梯度下降算法和神经网络的学习联系起来，假设$w_k$和$b_l$表示我们神经网络中权重和偏置，即可得到：
$$
\Delta w_k = -\frac{\eta}{m}\sum_j\frac{\partial C_{X_j}}{\partial w_k}
\\\\
\Delta b_l = -\frac{\eta}{m}\sum_j\frac{\partial C_{X_j}}{\partial b_l}
$$

## 1.6 实现我们的网络来分类数字

&nbsp;&nbsp;&nbsp;&nbsp;在这里，我们将使用随机梯度下降算法和MNIST训练数据来编写一个识别手写数字的程序。

```python
import random

import numpy as np

class Network(object):

    def __init__(self, sizes):
        """
        网络的构造函数.
        :sizes: 各层神经元的数量,例如:[2,3,1].[list]
        """
        self.num_layers = len(sizes)  # 网络的层数
        self.sizes = sizes
        # 随机初始化偏置和权重并以Numpy矩阵列表的形式存储.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        :a: 网络给定的输入
        :return: 网络输入对应的输出
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        随机梯度下降.
        :training_data: 训练输入和其对应的期望输出[a list of tuples]
        :epochs: 迭代次数
        :mini_batch_size: 小批量数据的大小
        :eta: 学习速率
        :test_data: 验证数据
        """
        if test_data: n_test = len(test_data)  # 如果给出test_data,程序会在每次迭代之后评估网络
        n = len(training_data)
        for j in xrange(epochs):  # 在每个迭代期
            random.shuffle(training_data)  # 随机将训练数据打乱
            mini_batches = [  # 将数据分成多个大小适当的小批量数据
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)  # 对每一个mini_batch进行一次梯度下降
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """
        梯度下降.根据单次梯度下降的迭代来更新网络的权重和偏置.
        :mini_batch: 小批量数据 [a list of tuples]
        :eta: 学习速率
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)# 这个函数大部分工作由这行代码完成
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

def sigmoid(z):
    """S型函数."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

```

&nbsp;&nbsp;&nbsp;&nbsp;$w$表示矩阵，$w_{jk}$是连接第二层$k^{th}$神经元和第三层$j^{th}$神经元的权重。第三层神经元的激活向量是：
$$
{a}'=\sigma(wa+b)
$$
&nbsp;&nbsp;&nbsp;&nbsp;化成分量形式：
$$
\frac{1}{1+e^{-wa-b}}=\frac{1}{1 + exp(-\sum_j{w_j-b)}}
$$
&nbsp;&nbsp;&nbsp;&nbsp;`到这里就不在贴代码出来了，剩余的代码书上给的链接可以下载，并且里面的注释相当详细，可自行研究。(在读代码时有一些地方不必完全看懂，只要有一个概念，知道它是干嘛的就可以了，后期书上会讲到或者有需要可自己去查看相关资料)`