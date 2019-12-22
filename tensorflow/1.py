import tensorflow as tf

message = tf.constant('Welcome to the world of deep neural network!')

with tf.Session() as sess:
    print(sess.run(message).decode())


