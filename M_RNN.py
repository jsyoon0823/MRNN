'''
Jinsung Yoon (06/19/2018)
M-RNN Architecture
'''

#%% Packages
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops

#%% Main Function
def M_RNN (trainX, trainZ, trainM, trainT, testX, testZ, testM, testT):

    # Graph Initialization
    ops.reset_default_graph()
    
    #%% Parameters
    seq_length = len(trainX[0,:,0])
    feature_dim = len(trainX[0,0,:])
    hidden_dim = 10
    
    learning_rate = 0.01
    iterations = 1000

    #%% input place holders
    Y = tf.placeholder(tf.float32, [seq_length, None, 1])
    
    #%% Weights Initialization        
            
    class Bi_LSTM_cell(object):
    
        """
        Bi directional LSTM cell object which takes 3 arguments for initialization.
        input_size = Input Vector size
        hidden_layer_size = Hidden layer size
        target_size = Output vector size
        """
    
        def __init__(self, input_size, hidden_layer_size, target_size):
    
            # Initialization of given values
            self.input_size = input_size
            self.hidden_layer_size = hidden_layer_size
            self.target_size = target_size
    
            # Weights and Bias for input and hidden tensor for forward pass
            self.Wr = tf.Variable(tf.zeros([self.input_size, self.hidden_layer_size]))
            self.Ur = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))
            self.br = tf.Variable(tf.zeros([self.hidden_layer_size]))
    
            self.Wu = tf.Variable(tf.zeros([self.input_size, self.hidden_layer_size]))
            self.Uu = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))
            self.bu = tf.Variable(tf.zeros([self.hidden_layer_size]))
    
            self.Wh = tf.Variable(tf.zeros([self.input_size, self.hidden_layer_size]))
            self.Uh = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))
            self.bh = tf.Variable(tf.zeros([self.hidden_layer_size]))
    
            # Weights and Bias for input and hidden tensor for backward pass
            self.Wr1 = tf.Variable(tf.zeros([self.input_size, self.hidden_layer_size]))
            self.Ur1 = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))
            self.br1 = tf.Variable(tf.zeros([self.hidden_layer_size]))
    
            self.Wu1 = tf.Variable(tf.zeros([self.input_size, self.hidden_layer_size]))
            self.Uu1 = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))
            self.bu1 = tf.Variable(tf.zeros([self.hidden_layer_size]))
    
            self.Wh1 = tf.Variable(tf.zeros([self.input_size, self.hidden_layer_size]))
            self.Uh1 = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))
            self.bh1 = tf.Variable(tf.zeros([self.hidden_layer_size]))
    
            # Weights for output layers
            self.Wo = tf.Variable(tf.truncated_normal([self.hidden_layer_size * 2, self.target_size], mean=0, stddev=.01))
            self.bo = tf.Variable(tf.truncated_normal([self.target_size], mean=0, stddev=.01))
    
            # Placeholder for input vector with shape[batch, seq, embeddings]
            self._inputs = tf.placeholder(tf.float32, shape=[None, None, self.input_size], name='inputs')
    
            # Reversing the inputs by sequence for backward pass of the LSTM
            self._inputs_rev = tf.placeholder(tf.float32, shape=[None, None, self.input_size], name='inputs')
    
            # Processing inputs to work with scan function
            self.processed_input = process_batch_input_for_RNN(self._inputs)
    
            # For bacward pass of the LSTM
            self.processed_input_rev = process_batch_input_for_RNN(self._inputs_rev)
    
            '''
            Initial hidden state's shape is [1,self.hidden_layer_size]
            In First time stamp, we are doing dot product with weights to
            get the shape of [batch_size, self.hidden_layer_size].
            For this dot product tensorflow use broadcasting. But during
            Back propagation a low level error occurs.
            So to solve the problem it was needed to initialize initial
            hiddden state of size [batch_size, self.hidden_layer_size].
            So here is a little hack !!!! Getting the same shaped
            initial hidden state of zeros.
            '''
    
            self.initial_hidden = self._inputs[:, 0, :]
            self.initial_hidden = tf.matmul(self.initial_hidden, tf.zeros([input_size, hidden_layer_size]))
    
        # Function for Forward LSTM cell.
        def Lstm_f(self, previous_hidden_state, x):
            """
            This function takes previous hidden state
            and memory tuple with input and
            outputs current hidden state.
            """
    
            # R Gate
            r = tf.sigmoid(tf.matmul(x, self.Wr) + tf.matmul(previous_hidden_state, self.Ur) + self.br)
    
            # U Gate
            u = tf.sigmoid(tf.matmul(x, self.Wu) + tf.matmul(previous_hidden_state, self.Uu) + self.bu)
    
            # Final Memory cell
            c = tf.tanh(tf.matmul(x, self.Wh) + tf.matmul( tf.multiply(r, previous_hidden_state), self.Uh) + self.bh)
    
            # Current Hidden state
            current_hidden_state = tf.multiply( (1 - u), previous_hidden_state ) + tf.multiply( u, c )
    
            return current_hidden_state
    
    
        # Function for Forward LSTM cell.
        def Lstm_b(self, previous_hidden_state, x):
            """
            This function takes previous hidden
            state and memory tuple with input and
            outputs current hidden state.
            """
    
            r = tf.sigmoid(tf.matmul(x, self.Wr1) + tf.matmul(previous_hidden_state, self.Ur1) + self.br1)
    
            # U Gate
            u = tf.sigmoid(tf.matmul(x, self.Wu1) + tf.matmul(previous_hidden_state, self.Uu1) + self.bu1)
    
            # Final Memory cell
            c = tf.tanh(tf.matmul(x, self.Wh1) + tf.matmul( tf.multiply(r, previous_hidden_state), self.Uh1) + self.bh1)
    
            # Current Hidden state
            current_hidden_state = tf.multiply( (1 - u), previous_hidden_state ) + tf.multiply( u, c )
    
            return current_hidden_state
            
    
        # Function to get the hidden and memory cells after forward pass
        def get_states_f(self):
            """
            Iterates through time/ sequence to get all hidden state
            """
    
            # Getting all hidden state throuh time
            all_hidden_states = tf.scan(self.Lstm_f, self.processed_input, initializer=self.initial_hidden, name='states')
    
            return all_hidden_states
    
        # Function to get the hidden and memory cells after backward pass
        def get_states_b(self):
            """
            Iterates through time/ sequence to get all hidden state
            """
    
            all_hidden_states = self.get_states_f()
    
            # Reversing the hidden and memory state to get the final hidden and
            # memory state
            last_hidden_states = all_hidden_states[-1]
    
            # For backward pass using the last hidden and memory of the forward
            # pass
            initial_hidden = last_hidden_states
    
            # Getting all hidden state throuh time
            all_hidden_memory_states = tf.scan(self.Lstm_b, self.processed_input_rev, initializer=initial_hidden, name='states')
    
            # Now reversing the states to keep those in original order
            #all_hidden_states = tf.reverse(all_hidden_memory_states, [False, True, False])
    
            return all_hidden_states
    
        # Function to concat the hiddenstates for backward and forward pass
        def get_concat_hidden(self):
    
            # Getting hidden and memory for the forward pass
            all_hidden_states_f = self.get_states_f()
    
            # Getting hidden and memory for the backward pass
            all_hidden_states_b = self.get_states_b()
    
            # Concating the hidden states of forward and backward pass
            concat_hidden = tf.concat([all_hidden_states_f, all_hidden_states_b],2)
    
            return concat_hidden
    
        # Function to get output from a hidden layer
        def get_output(self, hidden_state):
            """
            This function takes hidden state and returns output
            """
            output = tf.nn.sigmoid(tf.matmul(hidden_state, self.Wo) + self.bo)
    
            return output
    
        # Function for getting all output layers
        def get_outputs(self):
            """
            Iterating through hidden states to get outputs for all timestamp
            """
            all_hidden_states = self.get_concat_hidden()
    
            all_outputs = tf.map_fn(self.get_output, all_hidden_states)
    
            return all_outputs
    
    
    # Function to convert batch input data to use scan ops of tensorflow.
    def process_batch_input_for_RNN(batch_input):
        """
        Process tensor of size [5,3,2] to [3,5,2]
        """
        batch_input_ = tf.transpose(batch_input, perm=[2, 0, 1])
        X = tf.transpose(batch_input_)
    
        return X

        
    # Initializing rnn object
    rnn = Bi_LSTM_cell(3, hidden_dim, 1)
    
    # Getting all outputs from rnn
    outputs = rnn.get_outputs()

    # reshape out for sequence_loss
    loss = tf.sqrt(tf.reduce_mean(tf.square(outputs - Y)))

    #
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # RMSE
    targets = tf.placeholder(tf.float32, [None, seq_length, feature_dim])
    predictions = tf.placeholder(tf.float32, [None, seq_length, feature_dim])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

    # Output Initialization
    final_results_train = np.zeros([len(trainX), seq_length, feature_dim])
    final_results_test = np.zeros([len(testX), seq_length, feature_dim])

    # Sessions
    sess = tf.Session()

    for f in range(feature_dim):
        sess.run(tf.global_variables_initializer())
        
        # Training step
        for i in range(iterations):

            Input_Temp = np.reshape(np.concatenate((trainZ[:,:,f],trainM[:,:,f],trainT[:,:,f]),0),[len(trainX), seq_length, 3]) 
            Input_Temp_Rev = np.flip(Input_Temp, 1)
            
            Input = np.zeros([len(trainX), seq_length, 3]) 
            Input[:,1:,:] = Input_Temp[:,:2,:] 
            
            Input_Rev = np.zeros([len(trainX), seq_length, 3]) 
            Input_Rev[:,1:,:] = Input_Temp_Rev[:,:2,:] 
            
            _, step_loss = sess.run([train, loss], feed_dict={Y: np.reshape(trainX[:,:,f],[seq_length, len(trainX), 1]),
                                    rnn._inputs: Input, rnn._inputs_rev: Input_Rev})
            
            if i % 100 == 0:
                print("[step: {}] loss: {}".format(i, step_loss))

        # Train step

        Input_Temp = np.reshape(np.concatenate((trainZ[:,:,f],trainM[:,:,f],trainT[:,:,f]),0),[len(trainX), seq_length, 3]) 
        Input_Temp_Rev = np.flip(Input_Temp, 1)
            
        Input = np.zeros([len(trainX), seq_length, 3]) 
        Input[:,1:,:] = Input_Temp[:,:2,:] 
            
        Input_Rev = np.zeros([len(trainX), seq_length, 3]) 
        Input_Rev[:,1:,:] = Input_Temp_Rev[:,:2,:] 

        train_predict = sess.run(outputs, feed_dict={rnn._inputs: Input, rnn._inputs_rev: Input_Rev})
        final_results_train[:,:,f] = np.reshape(train_predict, [len(trainX), seq_length])
        
        # Test step
        
        Input_Temp = np.reshape(np.concatenate((testZ[:,:,f],testM[:,:,f],testT[:,:,f]),0),[len(testX), seq_length, 3]) 
        Input_Temp_Rev = np.flip(Input_Temp, 1)
            
        Input = np.zeros([len(testX), seq_length, 3]) 
        Input[:,1:,:] = Input_Temp[:,:2,:] 
            
        Input_Rev = np.zeros([len(testX), seq_length, 3]) 
        Input_Rev[:,1:,:] = Input_Temp_Rev[:,:2,:]       
        
        test_predict = sess.run(outputs, feed_dict={rnn._inputs: Input, rnn._inputs_rev: Input_Rev})
        final_results_test[:,:,f] = np.reshape(test_predict, [len(testX), seq_length])
    
    
    #%%

    # Change the data structure
    Train_No = len(trainX[:,0,0])
    Test_No = len(testX[:,0,0])
    Seq_No = len(trainX[0,:,0])
    Dim_No = len(trainX[0,0,:])
            
    rec_trainX = final_results_train * (1-trainM) + trainX * trainM
    rec_testX = final_results_test * (1-testM) + testX * testM

    col_trainX = np.reshape(trainX, [Train_No * Seq_No, Dim_No])
    col_rec_trainX = np.reshape(rec_trainX, [Train_No * Seq_No, Dim_No])
    
    col_testX = np.reshape(testX, [Test_No * Seq_No, Dim_No])
    col_rec_testX = np.reshape(rec_testX, [Test_No * Seq_No, Dim_No])
    
    col_trainM = np.reshape(trainM, [Train_No * Seq_No, Dim_No])
    col_testM = np.reshape(testM, [Test_No * Seq_No, Dim_No])

    # train Parameters
    ops.reset_default_graph()    
    
    feature_dim = Dim_No
    
    learning_rate = 0.01
    iterations = 10000

    hidden_no = int(Dim_No/2)

    # input place holders
    X = tf.placeholder(tf.float32, [None, feature_dim])
    Z = tf.placeholder(tf.float32, [None, feature_dim * 2])

    # build a FC network
    W1 = tf.get_variable("W1", shape=[feature_dim * 2, hidden_no],initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([hidden_no]))
    L1 = tf.nn.relu(tf.matmul(Z, W1) + b1)

    W2 = tf.get_variable("W2", shape=[hidden_no, hidden_no],initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([hidden_no]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

    W3 = tf.get_variable("W3", shape=[hidden_no, feature_dim],initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([feature_dim]))
    hypothesis = tf.matmul(L2, W3) + b3

    outputs = tf.nn.sigmoid(hypothesis)
    
    # reshape out for sequence_loss
    loss = tf.sqrt(tf.reduce_mean(tf.square(outputs - X)) )

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # RMSE
    targets = tf.placeholder(tf.float32, [None, feature_dim])
    predictions = tf.placeholder(tf.float32, [None, feature_dim])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

    # Sessions
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
        
    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={X: col_trainX, Z: np.concatenate((col_rec_trainX, col_trainM),1)})
        
        if i % 100 == 0:
            print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
    train_predict = sess.run(outputs, feed_dict={Z: np.concatenate((col_rec_trainX, col_trainM),1)})
    test_predict = sess.run(outputs, feed_dict={Z: np.concatenate((col_rec_testX, col_testM),1)})
        
    rmse_val = sess.run(rmse, feed_dict={targets: col_testX, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))
    
    output_train_temp = np.reshape(train_predict,[Train_No, Seq_No, feature_dim])
    output_test_temp = np.reshape(test_predict,[Test_No, Seq_No, feature_dim])

    output_train = output_train_temp * (1-trainM) + trainX * trainM
    output_test = output_test_temp * (1-testM) + testX * testM


    #%%    
    
    return [output_train, output_test]