import numpy as np
import tensorflow as tf


def get_model(global_step):
    x_train = np.genfromtxt('beer_data/x_train.csv', delimiter=' ')
    x_test = np.genfromtxt('beer_data/x_test.csv', delimiter=' ')
    y_train_ = np.genfromtxt('beer_data/y_train.csv', delimiter=' ', dtype=np.int32)
    y_test_ = np.genfromtxt('beer_data/y_test.csv', delimiter=' ', dtype=np.int32)

    print(x_train.shape)
    print(y_train_.shape)

    # Parameters
    learning_rate = 0.012
    batch_size = 50

    # Network Parameters
    n_hidden_1 = 50 # 1st layer number of neurons
    n_hidden_2 = 50 # 2nd layer number of neurons
    num_input = x_train.shape[1]
    num_samples_train = x_train.shape[0]
    num_samples_test = x_test.shape[0]
    num_classes = 176

    y_train = [np.zeros(num_classes) for i in range(num_samples_train)]
    for i in range(num_samples_train):
        y_train[i][int(y_train_[i]) - 1] = np.int32(1)

    y_test = [np.zeros(num_classes) for i in range(num_samples_test)]
    for i in range(num_samples_test):
        y_test[i][int(y_test_[i]) - 1] = np.int32(1)

    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder(np.int32, [None, num_classes])

    # Create model
    def neural_net(x):
        layer1 = tf.layers.dense(x, n_hidden_1, activation=tf.nn.relu)
        layer2 = tf.layers.dense(layer1, n_hidden_2, activation=tf.nn.relu)
        out_layer = tf.layers.dense(layer2, num_classes)
        return out_layer

    # Construct model
    logits = neural_net(X)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step = global_step)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    return X, Y, x_train, y_train, loss_op, train_op, accuracy


if __name__ == '__main__':
    X, Y, x_train, y_train, loss_op, train_op, accuracy = get_model()

    num_steps = 100
    display_step = 5

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for step in range(1, num_steps+1):
            batch_x, batch_y = x_train, y_train
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

        print("Optimization Finished!")

        # Calculate accuracy for MNIST test images
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: x_test,
                                          Y: y_test}))
