from blocks.bricks import (application, lazy, Initializable,
                           Tanh, Linear, MLP, Softmax, Rectifier)
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import (Bidirectional, SimpleRecurrent, LSTM,
                                     GatedRecurrent)
from blocks.initialization import IsotropicGaussian, Constant, Orthogonal, Uniform

from blocks.bricks.lookup import LookupTable
from blocks.utils import dict_union
from picklable_itertools.extras import equizip
from theano import tensor
from theano import scan
from sampler import gen_sample
import theano


class Generator(Initializable):
    """Either teacher forcing or free generator.
    """

    @lazy()
    def __init__(self, networks, dims, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.dims = dims
        self.networks = networks
        self.hid_linear_trans = [Fork([name for name in
                                            networks[i].apply.sequences
                                            if name != 'mask'],
                                           name='fork_{}'.format(i),
                                           prototype=Linear(), **kwargs)
                                      for i in range(len(networks))]

        self.out_linear_trans = Linear(name='out_linear', **kwargs)
        self.children = (networks +
                         self.hid_linear_trans +
                         [self.out_linear_trans])
        self.num_layers = len(networks)

    @application
    def apply(self, x, input_mask, *args, **kwargs):
        self.tf = kwargs.pop('tf', True)
        y_tm1 = kwargs.pop('targets')

        if not self.tf:
            states_dim = (len(self.dims) - 2) * self.dims[1]
            states = tensor.zeros((states_dim,))[None, :]
            states = tensor.repeat(states, x.shape[1], 0)
            self.children.append(Softmax())
        else:
            x = tensor.concatenate([x, y_tm1], axis=2).astype('float32')
            raw_states = x

        if self.tf:
            all_raw_states = []
            for i in range(self.num_layers):
                transformed_x = self.hid_linear_trans[i].apply(
                        raw_states)
                raw_states, raw_cells = self.networks[i].apply(transformed_x,
                                                       mask=input_mask,
                                                       *args, **kwargs)
                all_raw_states.append(raw_states)
            encoder_out = self.out_linear_trans.apply(raw_states)
            raw_states = tensor.concatenate(all_raw_states,
                                            axis=2).astype('float32')
            return encoder_out, raw_states

        else:
            def one_step(input_, mask_, label_tm1, states_tm1, cells_tm1):

                cells_update = []
                states_update = []
                raw_states = tensor.concatenate([input_, label_tm1],
                                                axis=1).astype('float32')

                for i in range(self.num_layers):
                    transformed_x = self.hid_linear_trans[i].apply(
                        raw_states)
                    if states_tm1.shape[1] != self.dims[1]:
                        shape = self.dims[1]
                        states_tm1_ = states_tm1[:, i*shape: (i+1)*shape]

                    if cells_tm1.shape[1] != self.dims[1]:
                        shape = self.dims[1]
                        cells_tm1_ = cells_tm1[:, i*shape: (i+1)*shape]

                    raw_states, raw_cells = self.networks[i].apply(transformed_x,
                                                                   states_tm1_,
                                                                   cells_tm1_,
                                                                   mask=mask_,
                                                                   iterate=False,
                                                                   *args, **kwargs)
                    cells_update.append(raw_cells)
                    states_update.append(raw_states)

                cells_out = tensor.concatenate(cells_update, axis=1).astype('float32')
                states_out = tensor.concatenate(states_update, axis=1).astype('float32')
                encoder_out = self.out_linear_trans.apply(raw_states)
                encoder_dist = self.children[-1].apply(encoder_out)
                labels_out = gen_sample(encoder_dist, argmax=False)
                return labels_out, states_out, cells_out, encoder_out

            each_cells = []
            for i in range(self.num_layers):
                each_cells.append(self.networks[i].initial_cells[None, :])
            cells = tensor.concatenate(each_cells, axis=1).astype('float32')
            cells = tensor.repeat(cells, x.shape[1], 0)
            ([next_labels, next_states,
                    next_cells, soft_out], scan_updates) = theano.scan(fn=one_step,
                                                 sequences=[x, input_mask],
                                                 outputs_info=[y_tm1, states,
                                                               cells, None])
            return next_labels, next_states, next_cells, soft_out, scan_updates


    def _push_allocation_config(self):
        if not len(self.dims) - 2 == self.num_layers:
            raise ValueError

        self.hid_linear_trans[0].input_dim = self.dims[0]
        self.hid_linear_trans[0].output_dims = \
            [self.networks[0].get_dim(name) for
             name in self.hid_linear_trans[0].input_names]

        for network, input_dim, layer in \
                equizip(self.networks[1:],
                        self.dims[1: -2],
                        self.hid_linear_trans[1:]):
            layer.input_dim = input_dim
            layer.output_dims = \
                [network.get_dim(name) for
                 name in layer.input_names]
            layer.use_bias = self.use_bias
        self.out_linear_trans.input_dim = self.dims[-2]
        self.out_linear_trans.output_dim = self.dims[-1]
        self.out_linear_trans.use_bias = self.use_bias


class MultiLayerEncoder(Initializable):
    """Stacked Bidirectional RNN.
    Parameters
    ---------
    networks : a list of instance of :class:`BidirectionalGraves`
    dims: a list of dimensions from the first network state to the last one.
    """

    @lazy()
    def __init__(self, networks, dims, **kwargs):
        super(MultiLayerEncoder, self).__init__(**kwargs)
        self.dims = dims
        self.networks = networks

        self.hid_linear_trans_forw = [Fork([name for name in
                                            networks[i].prototype.apply.sequences
                                            if name != 'mask'],
                                           name='fork_forw_{}'.format(i),
                                           prototype=Linear(), **kwargs)
                                      for i in range(len(networks))]

        self.hid_linear_trans_back = [Fork([name for name in
                                            networks[i].prototype.apply.sequences
                                            if name != 'mask'],
                                           name='fork_back_{}'.format(i),
                                           prototype=Linear(), **kwargs)
                                      for i in range(len(networks))]

        self.out_linear_trans = Linear(name='out_linear', **kwargs)
        self.mlp = MLP(activations=[Rectifier(), Rectifier(), Rectifier()],
                       dims=[self.dims[1]*2, self.dims[1], self.dims[1],
                             self.dims[1]],
                       weights_init=Uniform(width=.2),
                       biases_init=Constant(0.))
        self.children = (networks +
                         self.hid_linear_trans_forw +
                         self.hid_linear_trans_back +
                         [self.mlp, self.out_linear_trans])
        self.num_layers = len(networks)

    @application
    def apply(self, x, input_mask, *args, **kwargs):
        raw_states = x
        for i in range(self.num_layers):
            transformed_x_forw = self.hid_linear_trans_forw[i].apply(
                raw_states, as_dict=True)
            transformed_x_back = self.hid_linear_trans_back[i].apply(
                raw_states, as_dict=True)
            raw_states = self.networks[i].apply(transformed_x_forw,
                                                transformed_x_back,
                                                input_mask,
                                                *args, **kwargs)
        encoder_out = self.out_linear_trans.apply(self.mlp.apply(raw_states))
        return encoder_out

    def _push_allocation_config(self):
        if not len(self.dims) - 2 == self.num_layers:
            raise ValueError

        self.hid_linear_trans_forw[0].input_dim = self.dims[0]
        self.hid_linear_trans_back[0].input_dim = self.dims[0]
        self.hid_linear_trans_forw[0].output_dims = \
            [self.networks[0].prototype.get_dim(name) for
             name in self.hid_linear_trans_forw[0].input_names]
        self.hid_linear_trans_back[0].output_dims = \
            [self.networks[0].prototype.get_dim(name) for
             name in self.hid_linear_trans_back[0].input_names]

        for network, input_dim, layer_forw, layer_back in \
                equizip(self.networks[1:],
                        self.dims[1: -2],
                        self.hid_linear_trans_forw[1:],
                        self.hid_linear_trans_back[1:]):
            layer_forw.input_dim = input_dim * 2
            layer_forw.output_dims = \
                [network.prototype.get_dim(name) for
                 name in layer_forw.input_names]
            layer_forw.use_bias = self.use_bias
            layer_back.input_dim = input_dim * 2
            layer_back.output_dims = \
                [network.prototype.get_dim(name) for
                 name in layer_back.input_names]
            layer_back.use_bias = self.use_bias
        #self.out_linear_trans.input_dim = self.dims[-2] * 2
        self.out_linear_trans.input_dim = self.dims[-2]
        self.out_linear_trans.output_dim = self.dims[-1]
        self.out_linear_trans.use_bias = self.use_bias

class GeneratorTest(Initializable):
    @lazy()
    def __init__(self, networks, dims, **kwargs):
        super(GeneratorTest, self).__init__(**kwargs)
        self.dims = dims
        self.networks = networks
        self.hid_linear_trans = [Fork([name for name in
                                            networks[i].apply.sequences
                                            if name != 'mask'],
                                           name='fork_{}'.format(i),
                                           prototype=Linear(), **kwargs)
                                      for i in range(len(networks))]

        self.out_linear_trans = Linear(name='out_linear', **kwargs)
        self.children = (networks +
                         self.hid_linear_trans +
                         [self.out_linear_trans])
        self.num_layers = len(networks)

    @application
    def apply(self, x, input_mask, *args, **kwargs):
        states = kwargs.pop('states', None)
        cells = kwargs.pop('cells', None)
        y_tm1 = kwargs.pop('targets', None)
        x = tensor.concatenate([x, y_tm1], axis=1)
        raw_states = x
        cells_update = []
        states_update = []

        for i in range(self.num_layers):
            transformed_x = self.hid_linear_trans[i].apply(
                raw_states)
            if states.shape[0] != transformed_x.shape[1] // 4:
                shape = transformed_x.shape[1] // 4
                states_ = states[:, i*shape: (i+1)*shape]

            if cells.shape[0] != transformed_x.shape[1] // 4:
                shape = transformed_x.shape[1] // 4
                cells_ = cells[:, i*shape: (i+1)*shape]

            raw_states, raw_cells = self.networks[i].apply(transformed_x,
                                                           states_,
                                                           cells_,
                                                           mask=input_mask,
                                                           iterate=False,
                                                           *args, **kwargs)
            cells_update.append(raw_cells)
            states_update.append(raw_states)
        cells_out = tensor.concatenate(cells_update, axis=1)
        states_out = tensor.concatenate(states_update, axis=1)
        encoder_out = self.out_linear_trans.apply(raw_states)
        return encoder_out, states_out, cells_out

    def _push_allocation_config(self):
        if not len(self.dims) - 2 == self.num_layers:
            raise ValueError

        self.hid_linear_trans[0].input_dim = self.dims[0]
        self.hid_linear_trans[0].output_dims = \
            [self.networks[0].get_dim(name) for
             name in self.hid_linear_trans[0].input_names]

        for network, input_dim, layer in \
                equizip(self.networks[1:],
                        self.dims[1: -2],
                        self.hid_linear_trans[1:]):
            layer.input_dim = input_dim
            layer.output_dims = \
                [network.get_dim(name) for
                 name in layer.input_names]
            layer.use_bias = self.use_bias
        self.out_linear_trans.input_dim = self.dims[-2]
        self.out_linear_trans.output_dim = self.dims[-1]
        self.out_linear_trans.use_bias = self.use_bias
