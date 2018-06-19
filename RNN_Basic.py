'''
Jinsung Yoon (06/19/2018)
Prediction with LSTM
'''

#%% Packages
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops

#%% Main Function
def RNN_Basic (trainX, trainY, testX):

    # Reset the graph
    ops.reset_default_graph()
    
    #%% Parameters
    seq_length = 3
    data_dim = 112
    hidden_dim = 20
    output_dim = 1
    learning_rate = 0.01
    iterations = 10000
    Train_No = len(trainX[:,0,0])
    Test_No = len(testX[:,0,0])
    
    #%% Networks
    # input place holders
    # 1. Feature
    X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    # 2. Label   
    Y = tf.placeholder(tf.float32, [None, seq_length])
    # 3. Time    
    T = tf.placeholder(tf.float32, [None, 1])

    #%% Build a LSTM network
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    # FC layer
    X_for_fc = tf.reshape(outputs, [-1, hidden_dim])
    outputs = tf.contrib.layers.fully_connected(inputs=X_for_fc, num_outputs=output_dim, activation_fn=tf.sigmoid)
    
    # reshape out for sequence_loss
    outputs = tf.reshape(outputs, [tf.size(T), seq_length])

    # MSE Loss
    loss = tf.reduce_mean(tf.square(outputs - Y))

    # Optimization
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    #%% Session 
    sess = tf.Session()    
    sess.run(tf.global_variables_initializer())

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY, T:np.zeros([Train_No,1])})
        
        if i % 100 == 0:           
            print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(outputs, feed_dict={X: testX, T:np.zeros([Test_No,1])})
            
    return test_predict
