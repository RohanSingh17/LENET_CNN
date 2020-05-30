import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

tf=tf.compat.v1
tf.disable_eager_execution()

def display_random(x,y):
    print('Label: ',y)
    plt.imshow(x,cmap='gray')
    plt.show()

def lenet(x):
    weights={
        'wc1': tf.Variable(tf.random.truncated_normal([5, 5, 1, 6],mean=0,stddev=0.1), tf.float32),
        'wc2': tf.Variable(tf.random.truncated_normal([5, 5, 6, 16],mean=0,stddev=0.1), tf.float32),
        'wc3': tf.Variable(tf.random.truncated_normal([400,120],mean=0,stddev=0.1), tf.float32),
        'wc4': tf.Variable(tf.random.truncated_normal([120,84],mean=0,stddev=0.1), tf.float32),
        'wc5': tf.Variable(tf.random.truncated_normal([84,10],mean=0,stddev=0.1), tf.float32)
    }

    bias={
        'b1': tf.Variable(tf.zeros([6])),
        'b2': tf.Variable(tf.zeros([16])),
        'b3': tf.Variable(tf.zeros([120])),
        'b4': tf.Variable(tf.zeros([84])),
        'b5': tf.Variable(tf.zeros([10]))
    }

    stride=[1,1,1,1]
    conv1=tf.nn.conv2d(x,weights['wc1'],stride,padding='VALID')
    conv1=tf.nn.bias_add(conv1,bias['b1'])
    conv1_relu=tf.nn.relu(conv1)

    k=2
    conv1_pool=tf.nn.max_pool(conv1_relu,[1,2,2,1],[1,2,2,1],padding='VALID')

    conv2=tf.nn.conv2d(conv1_pool,weights['wc2'],stride,padding='VALID')
    conv2=tf.nn.bias_add(conv2,bias['b2'])
    conv2_relu=tf.nn.relu(conv2)

    conv2_pool=tf.nn.max_pool(conv2_relu,[1,2,2,1],[1,2,2,1],padding='VALID')

    conv2_flatten=tf.reshape(conv2_pool,[-1,400])

    conv3=tf.add(tf.matmul(conv2_flatten,weights['wc3']),bias['b3'])
    conv3_relu=tf.nn.relu(conv3)

    conv4=tf.add(tf.matmul(conv3_relu,weights['wc4']),bias['b4'])
    conv4_relu=tf.nn.relu(conv4)

    logits=tf.add(tf.matmul(conv4_relu,weights['wc5']),bias['b5'])

    return logits

def evaluate_accuracy(x1,y1):
    num_examples = len(x1)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x= x1[offset:offset+batch_size]
        batch_y = y1[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, labels: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    #outputFeatureMap(x1, 'conv1_relu', activation_min=-1, activation_max=-1, plt_num=1)
    return total_accuracy / num_examples

    # correct_pred=tf.cast((tf.argmax(pred,1)==tf.argmax(y,1)),tf.float32)
    # accuracy=tf.reduce_mean(correct_pred)

    # return accuracy

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")


learning_rate=0.001
epoch=5
batch_size=128

mnist=tf.keras.datasets.mnist
(X_train,Y_train),(x_test, y_test)=mnist.load_data()

a1=X_train[0].shape[0]
a2=X_train[0].shape[1]
X_tr=np.array([np.reshape(X_train[i],(a1,a2,1)) for i in range(len(X_train))])
X_ts=np.array([np.reshape(x_test[i],(a1,a2,1)) for i in range(len(x_test))])

onehotencoder=OneHotEncoder(sparse=False)
y_train1=Y_train.reshape(-1,1)
y_train_labels1=onehotencoder.fit_transform(y_train1)

y_test1=y_test.reshape(-1,1)
y_test_labels=onehotencoder.fit_transform(y_test1)
print(X_train.shape)

X_train=np.pad(X_tr,((0,0),(2,2),(2,2),(0,0)),'constant')
x_test=np.pad(X_ts,((0,0),(2,2),(2,2),(0,0)),'constant')
print(X_train.shape)

x_train, x_valid, y_train_labels, y_valid_labels = train_test_split(X_train, y_train_labels1, test_size=0.1)
N=len(x_train)

rnd=np.random.randint(len(x_train))
#display_random(x_train[rnd],y_train[rnd])

x=tf.placeholder(tf.float32,(None,32,32,1))
labels=tf.placeholder(tf.float32,(None,10))

logits=lenet(x)
prediction=tf.nn.softmax(logits)
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)
cost=tf.reduce_mean(cross_entropy)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     valid_accuracy=[]
#     for i in range(epoch):
#         for start in range(0,N,batch_size):
#             train_batch=x_train[start:start+batch_size]
#             labels_batch=y_train_labels[start:start+batch_size]
#             sess.run(optimizer,feed_dict={x:train_batch,labels:labels_batch})
#
#         validation_acc=evaluate_accuracy(x_valid,y_valid_labels)
#         valid_accuracy.append(validation_acc)
#         print("EPOCH {}...".format(i+1))
#         print("Validation Accuracy = {:.3f}".format(validation_acc))
#
#     saver.save(sess, './lenet')
#     print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate_accuracy(x_test, y_test_labels)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

