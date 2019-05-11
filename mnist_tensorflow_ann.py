import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import random, pickle

glob_cost = []
glob_acc = []

file_one = open('cost_fib.pkl', 'wb')
file_two = open('accur_fib.pkl', 'wb')


def fib_init(X_val,Y_val):

    mul_fact = 0.001
    
    fib_series = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
    
    W = np.zeros((X_val, Y_val))
    
    for i in range(X_val):
        for j in range(Y_val):
            W[i][j] = random.choice(fib_series) * mul_fact
            if(random.uniform(0, 1)<0.5):
                W[i][j] = -W[i][j]
    
    return np.float32(W)
    
def fib_bias(X_val):
    mul_fact = 0.001
    fib_series = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    b = np.zeros(X_val)
    for i in range(X_val):
        b[i] = random.choice(fib_series) * mul_fact
        if(random.uniform(0, 1)<0.5):
            b[i] = -b[i]
    return np.float32(b)
    
def nn_example():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # Python optimisation variables
    learning_rate = 0.0161803399
    epochs = 300
    batch_size = 100
    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784
    x = tf.placeholder(tf.float32, [None, 784])
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 10])
    # now declare the weights connecting the input to the hidden layer
    # W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
    # b1 = tf.Variable(tf.random_normal([300]), name='b1')
    
    W1 = tf.Variable(fib_init(784,300), name='W1')
    b1 = tf.Variable(fib_bias(300), name='b1')
    
    # and the weights connecting the hidden layer to the output layer   
    # W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
    # b2 = tf.Variable(tf.random_normal([10]), name='b2')
    
    W2 = tf.Variable(fib_init(300, 10), name='W2')
    b2 = tf.Variable(fib_bias(10), name='b2')
    
    # calculate the output of the hidden layer
    hidden_out = tf.add(tf.matmul(x, W1), b1)
    hidden_out = tf.nn.relu(hidden_out)
    # now calculate the hidden layer output - in this case, let's use a softmax activated
    # output layer
    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))
    # now let's define the cost function which we are going to train the model on
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                                                  + (1 - y) * tf.log(1 - y_clipped), axis=1))
    # add an optimiser
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    # finally setup the initialisation operator
    init_op = tf.global_variables_initializer()
    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # add a summary to store the accuracy
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter('GoldenRatioProjects')
    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        total_batch = int(len(mnist.train.labels) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
            summary = sess.run(merged, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            # writer.add_summary(summary, epoch)
            glob_cost.append(avg_cost)
            print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
            glob_acc.append(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
        print("\nTraining complete!")
        # writer.add_graph(sess.graph)
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
        pickle.dump(glob_cost, file_one)
        pickle.dump(glob_acc, file_two)        

if __name__ == "__main__":
    nn_example() 
