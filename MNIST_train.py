# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from batches_func import batches

# Noramlize image data
def normalize(image_data):
    a=0.1
    b=0.9
    normlzd=a+(((b-a)*(image_data-np.min(image_data)))/(np.max(image_data)-np.min(image_data)))
    return normlzd

# Parameters
epoch=5
learning_rate = 0.05
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
batch_size = 128  # Decrease batch size if you don't have enough memory
display_step = 50

# Import MNIST data
#mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)
tf.compat.v1.disable_eager_execution()

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test, y_test)=mnist.load_data()

train_features=normalize(np.array([x_train[i].flatten() for i in range(len(x_train))]))
test_features=normalize(np.array([x_test[i].flatten() for i in range(len(x_test))]))

onehotencoder=OneHotEncoder(sparse=False)
y_labels=y_train.reshape(-1,1)
train_labels=onehotencoder.fit_transform(y_labels)

y_labels=y_test.reshape(-1,1)
test_labels=onehotencoder.fit_transform(y_labels)

weights=tf.Variable(tf.random.truncated_normal([n_input,n_classes]))
biases=tf.Variable(tf.zeros(n_classes))

features=tf.compat.v1.placeholder(tf.float32,[None,n_input])
labels=tf.compat.v1.placeholder(tf.float32,[None,n_classes])

logits=tf.add(tf.matmul(features,weights),biases)
prediction=tf.nn.softmax(logits)
cross_entropy=tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=logits)

cost=tf.reduce_mean(cross_entropy)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

is_correct_prediction=tf.equal(tf.compat.v1.argmax(prediction,1),tf.compat.v1.argmax(labels,1))
accuracy=tf.reduce_mean(tf.cast(is_correct_prediction,tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(epoch):

        for batch_i in range((len(x_train)//batch_size)):
            batch_start=batch_i*batch_size
            batch_features=train_features[batch_start:batch_start+batch_size]
            batch_labels=train_labels[batch_start:batch_start+batch_size]
            output=sess.run(optimizer,feed_dict={features:batch_features,labels:batch_labels})

            if not batch_i%display_step:
                print(sess.run(accuracy,feed_dict={features:train_features,labels:train_labels}))

    test_accuracy=sess.run(accuracy,feed_dict={features:test_features,labels:test_labels})
    print('Test Accuracy: {}'.format(test_accuracy))


# The features are already scaled and the data is shuffled
# train_features = mnist.train.images
# test_features = mnist.test.images
#
# train_labels = mnist.train.labels.astype(np.float32)
# test_labels = mnist.test.labels.astype(np.float32)

# Features and Labels
# features = tf.placeholder(tf.float32, [None, n_input])
# labels = tf.placeholder(tf.float32, [None, n_classes])
#
# # Weights & bias
# weights = tf.Variable(tf.random_normal([n_input, n_classes]))
# bias = tf.Variable(tf.random_normal([n_classes]))
#
# # Logits - xW + b
# logits = tf.add(tf.matmul(features, weights), bias)
#
# # Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#
# # Calculate accuracy
# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# # TODO: Set batch size
# batch_size = None
# assert batch_size is not None, 'You must set the batch size'
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     # TODO: Train optimizer on all batches
#     # for batch_features, batch_labels in ______
#     sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})
#
#     # Calculate accuracy for test dataset
#     test_accuracy = sess.run(
#         accuracy,
#         feed_dict={features: test_features, labels: test_labels})
#
# print('Test Accuracy: {}'.format(test_accuracy))