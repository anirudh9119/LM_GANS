import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng

class HiddenLayer: 

    def __init__(self, num_in, num_out, activation = None, batch_norm = False): 
        
        self.activation = activation
        self.batch_norm = batch_norm

        self.residual = (num_in == num_out)

        self.W = theano.shared(0.01 * np.random.normal(size = (num_in, num_out)).astype('float32'))
        self.b = theano.shared(0.0 * np.random.normal(size = (num_out)).astype('float32'))

    def get_output_for(self, input_raw):

        input = input_raw

        lin_output = T.dot(input, self.W) + self.b

        if self.batch_norm:
            lin_output = (lin_output - T.mean(lin_output, axis = 0, keepdims = True)) / (1.0 + T.std(lin_output, axis = 0, keepdims = True))
            lin_output += self.b

        self.out_store = lin_output

        if self.activation == None: 
            activation = lambda x: x
        elif self.activation == "relu": 
            activation = lambda x: T.maximum(0.0, x)
        elif self.activation == "exp": 
            activation = lambda x: T.exp(x)
        elif self.activation == "tanh":
            activation = lambda x: T.tanh(x)
        elif self.activation == 'softplus':
            activation = lambda x: T.nnet.softplus(x)
        else: 
            raise Exception("Activation not found")

        out = activation(lin_output)

        if self.residual:
            out += input

        return out


    def getParams(self):
        return [self.W, self.b]

