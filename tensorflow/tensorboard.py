import tensorflow as tf

w = tf.Variable([.3],name='w',dtype=tf.float32)
b = tf.Variable([-.3],name='b',dtype=tf.float32)
x = tf.placeholder(name='x',dtype=tf.float32)
y = w * x + b

with tf.Session() as tfs:
    tfs.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('tflogs',tfs.graph)
    print('run(y,{x:3}):',tfs.run(y,feed_dict={x:3}))
