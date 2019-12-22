import tensorflow as tf

sess = tf.InteractiveSession()

I_matrix = tf.eye(5)###声明一个5*5的恒等矩阵
#print(I_matrix.eval())###.eval()可以显示出结果

X = tf.Variable(tf.eye(10))###变量表示
X.initializer.run()###初始化
#print(X.eval())

A = tf.Variable(tf.random_normal([5,10]))
A.initializer.run()

product = tf.matmul(A,X)###矩阵相乘
#print(product.eval())

b = tf.Variable(tf.random_uniform([5,10],0,2,dtype=tf.int32))
b.initializer.run()
#print(b.eval())
b_new = tf.cast(b,dtype=tf.float32)###转换数据类型
#print(b_new.eval())

t_sum = tf.add(product,b_new)###矩阵相加
t_sub = product - b_new### 矩阵相减
print("A*X + b\n",t_sum.eval())
print("A*X - b\n",t_sub.eval())



