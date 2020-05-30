import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#
# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.compat.v1.Session()
# print(sess.run(hello))

# import tensorflow as tf
# msg = tf.constant('Hello, TensorFlow!')
# tf.print(msg)

# Create TensorFlow object called hello_constant
hello_constant = tf.constant('Hello World!')
with tf.compat.v1.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)