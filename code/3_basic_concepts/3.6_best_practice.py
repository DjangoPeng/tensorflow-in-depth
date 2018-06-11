# -*- coding=utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 打印日志的步长
log_step = 50
# ================ 1.定义超参数 ================
# 学习率
learning_rate = 0.01
# 最大训练步数
max_train_steps = 1000
# ================ 2.输入数据 ================
# 构造训练数据
train_X = np.array([[3.3],[4.4],[5.5],[6.71],[6.93],[4.168],[9.779],[6.182],[7.59],[2.167],[7.042],[10.791],[5.313],[7.997],[5.654],[9.27],[3.1]], dtype=np.float32)
train_Y = np.array([[1.7],[2.76],[2.09],[3.19],[1.694],[1.573],[3.366],[2.596],[2.53],[1.221],[2.827],[3.465],[1.65],[2.904],[2.42],[2.94],[1.3]], dtype=np.float32)
total_samples = train_X.shape[0]
# ================ 3.构建模型 ================
# 输入数据
X = tf.placeholder(tf.float32, [None, 1])
# 模型参数
W = tf.Variable(tf.random_normal([1, 1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")
# 推理值
Y = tf.matmul(X, W) + b
# ================ 4.定义损失函数 ================
# 实际值
Y_ = tf.placeholder(tf.float32, [None, 1])
# 均方差
loss = tf.reduce_sum(tf.pow(Y-Y_, 2))/(2*total_samples)
# ================ 5.创建优化器 ================
# 随机梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# ================ 6.定义单步训练操作 ================
# 最小化损失值
train_op = optimizer.minimize(loss)
# ================ 7.创建会话 ================
with tf.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer()) 
# ================ 8.迭代训练 ================
    print("Start training:")
    for step in xrange(max_train_steps):
        sess.run(train_op, feed_dict={X: train_X, Y_: train_Y})
        # 每隔log_step步打印一次日志
        if step % log_step == 0:
            c = sess.run(loss, feed_dict={X: train_X, Y_:train_Y})
            print("Step:%d, loss==%.4f, W==%.4f, b==%.4f" % 
                    (step, c, sess.run(W), sess.run(b)))
    # 计算训练完毕的模型在训练集上的损失值，作为指标输出
    final_loss = sess.run(loss, feed_dict={X: train_X, Y_: train_Y})
    # 计算训练完毕的模型参数W和b
    weight, bias = sess.run([W, b])
    print("Step:%d, loss==%.4f, W==%.4f, b==%.4f" % 
            (max_train_steps, final_loss, sess.run(W), sess.run(b)))
    print("Linear Regression Model: Y==%.4f*X+%.4f" % (weight, bias))
# ================ 模型可视化 ================
    # 初始化Matplotlib后端
    %matplotlib
    # 根据训练数据X和Y，添加对应的红色圆点
    plt.plot(train_X, train_Y, 'ro', label='Training data')
    # 根据模型参数和训练数据，添加蓝色（缺省色）拟合直线
    plt.plot(train_X, weight * train_X + bias, label='Fitted line')
    # 添加图例说明
    plt.legend()
    # 画出上面定义的图案
    plt.show()

'''
输出：
Start training:
Step:0, loss==2.8679, W==0.0054, b==0.0411
Step:50, loss==0.1045, W==0.3457, b==0.1317
Step:100, loss==0.1013, W==0.3402, b==0.1710
Step:150, loss==0.0985, W==0.3350, b==0.2080
Step:200, loss==0.0961, W==0.3301, b==0.2428
Step:250, loss==0.0939, W==0.3254, b==0.2755
Step:300, loss==0.0919, W==0.3211, b==0.3064
Step:350, loss==0.0902, W==0.3170, b==0.3354
Step:400, loss==0.0887, W==0.3131, b==0.3627
Step:450, loss==0.0874, W==0.3095, b==0.3884
Step:500, loss==0.0862, W==0.3061, b==0.4126
Step:550, loss==0.0851, W==0.3029, b==0.4353
Step:600, loss==0.0842, W==0.2999, b==0.4567
Step:650, loss==0.0833, W==0.2970, b==0.4769
Step:700, loss==0.0826, W==0.2944, b==0.4959
Step:750, loss==0.0820, W==0.2918, b==0.5137
Step:800, loss==0.0814, W==0.2895, b==0.5305
Step:850, loss==0.0809, W==0.2872, b==0.5463
Step:900, loss==0.0804, W==0.2851, b==0.5612
Step:950, loss==0.0800, W==0.2832, b==0.5752
Step:1000, loss==0.0797, W==0.2814, b==0.5881
Linear Regression Model: Y==0.2814*X+0.5881
'''