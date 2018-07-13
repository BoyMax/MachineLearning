# Tensorflow

## 深度神经网络模型


![network model](image/net_model.png)


## 机器学习的三要素
* 数据
* 算法
* 算力


##  神经网络搭建的八股
1. 数据集
2. 前向传播（构建计算图）
3. 反向传播（优化参数:损失函数、学习率、滑动平均、正则化） 
4. 训练网络（迭代训练网络）

### 网络结构
**前向传播**

>定义网络计算图，主要定义神经元间的大小、权重、偏移量

```
def forward(x, regularizer):
	w= 
	b= 
	y= 
	return y
	
def get_weight(shape, regularizer):
	w = tf.Variable()
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable()  
    return b
```

*tf.add\_to\_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w)) 表示把每一个w的正则化损失加入总损失losses中*


**反向传播**
>定义训练数据集（标签）、训练轮数、损失函数（均方误差[正则化]，交叉熵[正则化]）指数衰减学习率、滑动平均等

```
def backward():
    x = tf.placeholder()
    y_ = tf.placeholder()
    y = forward(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    loss_mse = tf.reduce_mean(tf.square(y-y_)) 

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, lables = tf.argmax(y_, 1))
    loss_cem = tf.reduce_mean(ce)

    #loss加入正则化(平方误差和 以及 交叉熵 两种方法)
    #正则化可提高泛化性
    loss = loss_mse + tf.add_n(tf.get_collection('losses'))
    loss = loss_cem + tf.add_n(tf.get_collection('losses'))

    #指数衰减学习率：加快优化效率
    learning_rate = tf.train.exponential_decay(LEARING_RATE_BASE, global_step, 总样本数/BATCH_SIZE, lEARNING_RATE_DECAY, staircase = True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name = 'train')

```

**训练网络**
>用会话来执行网络训练，进行多次迭代，优化网络参数，打印出一定轮数的参数值

```
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	
	for i in range (STEPS):
		sess.run(train_step, feed_dict= {x: , y_: })
		if i % 轮数 == 0:
			print 

```

## 示例
```
#coding:utf-8

#--------------------------------------------
#1. 生成训练数据集
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

SEED = 2 #随机种子

def generate_dateset():
    rdm = np.random.RandomState(SEED) #基于随机种子seed产生随机数
    X = rdm.randn(300, 2) #返回300行2列的矩阵 可表示300条数据，每条数据含有两个属性
    Y_ = [int(x0*x0 + x1*x1 <2) for (x0, x1) in X] #定义标签，两个属性的平方和小于2时，标签值为1，否则为0
    Y_c = [('red' if y else 'blue') for y in Y_] #定义标签值意义
    X = np.vstack(X).reshape(-1, 2) #-1表示跟随第二个元素计算， 第二个元素表示数据有多少列属性
    Y = np.vstack(Y_).reshape(-1, 1) #X有两列属性，Y只有一列

    return X, Y, Y_c


#--------------------------------------------
#2. 前向传播，定义神经网络

import tensorflow as tf

def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w)) 
    return w

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape = shape))  
    return b

def forward(x, regularizer):
    w1 = get_weight([2, 11], regularizer)
    b1 = get_bias([11])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1) #激活函数

    w2 = get_weight([11,1], regularizer)
    b2 = get_bias([1])
    y = tf.matmul(y1, w2) + b2  #输出层不激活

    return y
    

#--------------------------------------------
#3. 反向传播，定义神经网络参数，迭代优化

STEPS = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.01

def backward():
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    X, Y_, Y_c = generate_dateset()

    y = forward(x, REGULARIZER)

    global_step = tf.Variable(0, trainable=False)

    #loss加入正则化(平方误差和)
    loss_mse = tf.reduce_mean(tf.square(y-y_)) 
    loss = loss_mse + tf.add_n(tf.get_collection('losses'))

    #指数衰减学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 300/BATCH_SIZE, LEARNING_RATE_DECAY, staircase = True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)


    #训练过程
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
         
        for i in range (STEPS):
            start = (i*BATCH_SIZE) % 300
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict= {x: X[start:end], y_:Y_[start:end] })
            if i % 2000 == 0:
                loss_v = sess.run(loss, feed_dict= {x: X, y_:Y_})
                print 'After %d steps, loss is : %f' %(i, loss_v)
        xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = sess.run(y, feed_dict = {x : grid})
        probs = probs.reshape(xx.shape)

    plt.scatter(X[:,0], X[:,1], c = np.squeeze(Y_c))
    plt.contour(xx, yy, probs, levels = [.5])
    plt.show()


if __name__ == "__main__":
    backward()
        
```
