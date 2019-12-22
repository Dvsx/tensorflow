from tensorflow as tf

m = 10000#样本数量
n = 15#特征数量
p = 2#类别数量

#标准线性回归
X = tf.placeholder(tf.float32,name='X')
Y = tf.placeholder(tf.float32,name='Y')

w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)

Y_hat = x*w1 +w0#线性回归函数

loss = tf.square(Y - Y_hat,name='loss')#求平方

#多元线性回归(输入不止一个，输出只有一个)
X = tf.placeholder(tf.float32,name='X',shape=[m,n])
Y = tf.placeholder(tf.float32,name='Y')

w0 = tf.Variable(0.0)
w1 = tf.Variable(tf.random_normal([n,1]))

Y_hat = tf.matmul(X,w1) + w0

loss = tf.reduce_mean(tf.square(Y - Y_hat,name='loss'))

#逻辑回归
X = tf.placeholder(tf.float32,name='X',shape=[m,n])
Y = tf.placeholder(tf.float32,name='Y',shepe=[m,p])

w0 = tf.Variable(tf.zeros([1,p]),name='bias')
w1 = tf.Variable(tf.random_normal([n,1]),name='weights')

Y_hat = tf.matmul(X,w1) + w0

entropy = tf.nn.softmax_cross_entropy_with_logits(Y_hat,Y)#先sofmax再交叉熵
loss = tf.reduce_mean(entropy)#计算损失函数
