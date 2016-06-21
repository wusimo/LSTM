from Models import Models
import tensorflow as tf
from tensorflow.python.ops.constant_op import constant
from tensorflow.models.rnn import rnn, rnn_cell
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops



class LSTMModel(Models):
    
    def __init__(self,data_config,model_config):
        
        # data config
        
        self.n_steps = n_steps = data_config.n_steps
        self.n_inputs = n_inputs = data_config.n_inputs
        self.n_output = n_output = data_config.n_output
        self.n_train = n_train = data_config.n_train
        if not model_config.model =="LSTM":
            raise ValueError('Wrong Model')
        
        self.batch_size = batch_size = model_config.batch_size
        self.max_iter = model_config.max_iter
        self.learning_rate = model_config.learning_rate
        self.display_stride = model_config.display_stride
        
        self.n_hidden = n_hidden = model_config.n_hidden
        self.n_dense = n_dense = model_config.n_dense
        
        init_scale = 0.08           #Initial scale for the states
        
        #the architecture
        
        #from ruiming's code(which is hard to understand)
        #self.x = tf.placeholder(tf.float32,shape = [None,n_steps,n_inputs],name = 'input_data')
        #self.y_= tf.placeholder(tf.float32,shape = [None,n_output],name = 'output_data')# currently forecast only 1 data point, so n_output = 1
        
        self.x = tf.placeholder(tf.float32,shape = [None,n_steps,n_inputs],name = 'input_data')# None is for the batch size
        self.y_= tf.placeholder(tf.float32,shape = [None,n_output],name = 'output_data')
        
        #size of input channels
        in_channels = n_inputs
        #Used later on for drop_out. At testtime, we pass 1.0
        self.keep_prob = tf.placeholder("float", name = 'Drop_out_keep_prob')#dropout parameter
        # need to be configurable
        num_layers = 1
        
        weights = {
            'hidden' : tf.Variable(tf.random_normal([n_steps*n_inputs,n_hidden])),
            'dense' : tf.Variable(tf.random_normal([n_hidden,n_dense])),
            'out' : tf.Variable(tf.random_normal([n_dense,n_output]))
        }
        
        biasses = {
            'hidden' : tf.Variable(tf.random_normal([n_hidden])),
            'dense' : tf.Variable(tf.random_normal([n_dense])),
            'out' : tf.Variable(tf.random_normal([n_output]))
        }
        
        
        
        lstm_cell = rnn_cell.BasicLSTMCell(n_hidden,forget_bias = 1.0)
        if self.keep_prob < 1:
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            cell = rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
        initial_state = cell.zero_state(batch_size, tf.float32)
        
        #inputs = tf.expand_dims(self.x,3)
        
        # add a layer befor feeding to LSTM
        #_X = tf.reshape(self.x,[-1,n_steps*n_inputs])
        #inputs = tf.matmul(_X,weights['hidden'])+biasses['hidden']
        
        inputs = self.x
        initializer = tf.random_uniform_initializer(-init_scale,init_scale)
        #with tf.variable_scope("model", initializer=initializer):
        with tf.name_scope("LSTM") as scope:
            outputs = []
            state = initial_state
            with tf.variable_scope("LSTM_state"):

                for time_step in range(n_steps):
                    if time_step > 0: 
                        tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(inputs[:, time_step, :], state)
                    outputs.append(cell_output)

        with tf.name_scope("Softmax") as scope:
            
            with tf.variable_scope("Softmax_params"): 

                softmax_w = tf.Variable(tf.random_normal([n_hidden, n_output],name ="softmax_w"))
                softmax_b = tf.Variable(tf.random_normal([n_output],name ="softmax_b" ))
                #softmax_w = tf.get_variable("softmax_w", [n_hidden, n_output])                         
                #softmax_b = tf.get_variable("softmax_b", [n_output])                               
            

                
            logits = tf.matmul(cell_output, softmax_w) + softmax_b
        #Use sparse Softmax because we have mutually exclusive classes    
        #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,targets,name = 'Sparse_softmax')
            loss = tf.square(tf.sub(logits,self.y_))
        
            self.cost = tf.reduce_sum(loss) / batch_size
        #Pass on a summary to Tensorboard
            cost_summ = tf.scalar_summary('Cost',self.cost)
        # Calculate the accuracy
            #correct_prediction = tf.equal(tf.argmax(logits,1), targets)
            #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            #accuracy_summary = tf.scalar_summary("accuracy", accuracy)
        #self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_out, self.y_))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        print('here')
    
    def training(self,X_train,y_train):#,X_validation,y_validation,X_test,y_test):
        self.train_acc = []
        self.validation_acc = []
        self.test_acc = []
        self.train_loss = []
        self.validation_loss = []
        self.test_loss = []
        self.test_pred = []
        
        print("========================")
        print("Optimization..")
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            for itr in range(self.max_iter):
                
                ids = random.sample(range(self.n_train), self.batch_size)
                batch_x = X_train[ids]
                batch_y = y_train[ids]

                # Fit training using batch data
                sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y_: batch_y,self.keep_prob:0.9})

                
                if itr%100 == 0:
                    
                    result = sess.run(loss,feed_dict ={self.x: batch_x, self.y_: batch_y,self.keep_prob:0.9} )
                    print(result)
            saver = tf.train.Saver()
            save_path = saver.save(sess),"./model.ckpt"
                
            print("Optimization finished")
            print("==============================================")
            sess.close()
                
        
    