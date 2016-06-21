import numpy as np

class randConfig(object):
    np_rnd_seed = 3 # numpy
    rnd_seed =3 # random
    tf_seed = 3 # tensorflow
    
class dataConfig(object):
    n_steps = 6
    n_inputs = 4 # load,temp,weekday indicator, hour indicator
    n_train = 1000
    n_validation = 200
    n_test = 200
    n_output = 1
    
class LSTMConfig(object):
    model = "LSTM"
    learning_rate = 1e-3
    batch_size = 300
    max_iter = 3000
    display_stride = 100
    n_hidden = 40 #number of units in LSTM
    n_dense = 64 # number of units in the dense layer


config = randConfig()
np.random.seed(config.np_rnd_seed)
random.seed(config.rnd_seed)
tf.set_random_seed(config.tf_seed)
