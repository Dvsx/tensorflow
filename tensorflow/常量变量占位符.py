import tensorflow as tf

#声明一个标量常量
t_1 = tf.constant(4)

#声明一个向量常量
t_2 = tf.constant([4,3,2])

#声明一个所有元素为零的张量
zero_t = tf.zeros([2,3],tf.int32)

#声明一个所有元素为一的张量
ones_t = tf.ones([2,3],tf.int32)

#生成从初值到终值等差排列的序列,等差的计算(stop-start)/(num-1)
range_t = tf.linspace(2.0,5.0,5)

#创建具有不同分布的随机张量
#1.创建具有一定均值和标准差形状为[M,N]的正态分布随机数组
t_random = tf.random_normal([2,3],mean=2.0,stddev=4,seed=12)


#变量
rand_t = tf.random_uniform([50,50],0,10,seed=0)
t_a = tf.Variable(rand_t)
t_b = tf.Variable(rand_t)

#权重和偏置
weights = tf.Variable(tf.random_normal([100,100],stddev=2))

#bias = tf.Variable(tf.zeros[100],name='biases')

#用变量来初始化另一个变量
weights2 = tf.Variable(weights.initialized_value(),name='w2')

#占位符
x = tf.placeholder("float")
y = 2 * x
data = tf.random_uniform([4,5],10)
###注意，如果在定义部分使用print语句，则只会输出张量的类型，想要得到相关的值，使用sess.run()
with tf.Session() as sess:
    x_data = sess.run(data)
    print(sess.run(y,feed_dict={x:x_data}))
sess.close()

###注意，tensoflow序列不可迭代
