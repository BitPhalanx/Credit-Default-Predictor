import tensorflow as tf
import numpy as np
import random

def main():
    # Training Parameters to Tweak
    num_nodes_layer1 = 200   # bad=150,300,600
    num_nodes_layer2 = 200
    # num_nodes_layer3 = 2000 #400
    num_of_batches = 500#1000
    batch_size = 1000
    lr = 1E-3

    print ("Train on predicting default on credit account next month...")

    data = []
    labels = []

    with open("credit_default_dataset.txt","r") as input_file:
        for line in input_file:
            if(len(line.strip()) == 0):
                continue
            full_data_line = line.strip().split(",")
            data_line = full_data_line[1:24]
            label_line = full_data_line[24]
            data.append(data_line)
            labels.append(label_line)

    dataset = list(zip(data, labels))
    random.shuffle(dataset)
    test_length = int(len(dataset) * 0.9)

    print("Training on {} sets of data".format(test_length))
    print("Testing on {} sets of data".format(int(len(dataset)) - test_length))
    train_dataset = dataset[:test_length]
    test_dataset = dataset[test_length:]

    x_size = 23
    out_size = 2

    # inputs needs to be type float for matmul to work...
    inputs = tf.placeholder("float", shape=[None, x_size])
    labels = tf.placeholder("int32", shape=[None])

    weights1 = tf.get_variable("weight1", shape=[x_size, num_nodes_layer1], initializer=tf.contrib.layers.xavier_initializer())
    bias1 = tf.get_variable("bias1", shape=[num_nodes_layer1], initializer=tf.constant_initializer(value=0.0))
    
    layer1 = tf.nn.relu(tf.matmul(inputs, weights1) + bias1)

    weights2 = tf.get_variable("weight2", shape=[num_nodes_layer1, num_nodes_layer2], initializer=tf.contrib.layers.xavier_initializer())
    bias2 = tf.get_variable("bias2", shape=[num_nodes_layer2], initializer=tf.constant_initializer(value=0.0))
    
    layer2 = tf.nn.relu(tf.matmul(layer1, weights2) + bias2)

    # weights3 = tf.get_variable("weight3", shape=[num_nodes_layer2, num_nodes_layer3], initializer=tf.contrib.layers.xavier_initializer())
    # bias3 = tf.get_variable("bias3", shape=[num_nodes_layer3], initializer=tf.constant_initializer(value=0.0))

    # layer3 = tf.nn.relu(tf.matmul(layer2, weights3) + bias3)

    # weights4 = tf.get_variable("weight4", shape=[num_nodes_layer3, out_size], initializer=tf.contrib.layers.xavier_initializer())
    # bias4 = tf.get_variable("bias4", shape=[out_size], initializer=tf.constant_initializer(value=0.0))

    # outputs = tf.matmul(layer3, weights4) + bias4

    weights3 = tf.get_variable("weight3", shape=[num_nodes_layer2, out_size], initializer=tf.contrib.layers.xavier_initializer())
    bias3 = tf.get_variable("bias3", shape=[out_size], initializer=tf.constant_initializer(value=0.0))
    
    outputs = tf.matmul(layer2, weights3) + bias3

    # Back prop

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels, out_size), logits=outputs))
    train = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)

    predictions = tf.argmax(tf.nn.softmax(outputs), axis=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Training loop, sample from training dataset
        for epoch in range(num_of_batches):
            batch = random.sample(train_dataset, batch_size)
            inputs_batch, labels_batch = zip(*batch)
            train_np_inputs = np.asarray(inputs_batch).astype(np.float)
            train_np_labels = np.asarray(labels_batch).astype(np.int64)
            # print(train_np_inputs[5])
            # print(train_np_labels[5])
            loss_output, prediction_output, _ = sess.run([loss, predictions, train], feed_dict={inputs: train_np_inputs, labels: train_np_labels})

        # Now test our network accuracy on test data
        test_batch = random.sample(test_dataset, 3000)
        test_inputs_batch, test_labels_batch = zip(*batch)
        test_np_inputs = np.asarray(test_inputs_batch).astype(np.float)
        test_np_labels = np.asarray(test_labels_batch).astype(np.int64)
        test_loss_output, test_prediction_output = sess.run([loss, predictions], feed_dict={inputs: test_np_inputs, labels: test_np_labels})
        # f = open('output.txt', 'a+')
        f = open('output.txt', 'w')
        myString = 'Credit default training results:\n\n'
        print(myString)
        f.write(myString)
        # convert to numpy for accuracy check
        numpy_pred = np.asarray(test_prediction_output)
        numpy_label = np.asarray(test_labels_batch).astype(np.int64)
        myString = 'Prediction output: \n' + np.array_str(numpy_pred)
        # print(myString)
        f.write(myString)
        # print("\n\n")
        f.write('\n\n')
        myString = 'Labels batch: \n' + np.array_str(numpy_label)
        # print(myString)
        f.write(myString)
        myString = '\nLoss: ' + np.array_str(test_loss_output)
        # print(myString + "\n\n")
        f.write(myString + '\n\n')
        accuracy = np.mean(numpy_label == numpy_pred)
        myString = 'Accuracy: ' + np.array_str(accuracy)
        print(myString + "\n\n")
        f.write(myString)
        f.write('\n\n\n')
        f.close

if __name__ == "__main__":
    main()
