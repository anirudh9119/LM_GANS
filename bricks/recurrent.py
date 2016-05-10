import copy
import numpy
from theano import tensor

from blocks.bricks import (application, Initializable, Tanh,
                           Logistic, lazy, Rectifier)
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.roles import add_role, BIAS, WEIGHT, INITIAL_STATE

from blocks.utils import shared_floatx, shared_floatx_nans, shared_floatx_zeros
from picklable_itertools.extras import equizip

class BidirectionalGraves(Initializable):
    """Alex Grave's Bidirectional network that applies transformations separately.

    This is the attention mechanism used by Alex Graves in (CITE).

    Parameters
    ----------
    prototype : instance of :class:`BaseRecurrent`
        A prototype brick from which the forward and backward bricks are
        cloned.
    Notes
    -----
    See :class:`.Initializable` for initialization parameters.
    """

    @lazy()
    def __init__(self, prototype, **kwargs):
        super(BidirectionalGraves, self).__init__(**kwargs)
        self.prototype = prototype
        self.children = [copy.deepcopy(prototype) for _ in range(2)]
        self.children[0].name = 'forward'
        self.children[1].name = 'backward'

    @application
    def apply(self, transformed_x_forw, transformed_x_back,
              input_mask, *args, **kwargs):
        """Applies forward and backward networks and returns a list of outputs."""
        kwargs_forw = kwargs.copy()
        kwargs_back = kwargs.copy()
        kwargs_forw.update(transformed_x_forw)
        kwargs_back.update(transformed_x_back)
        forward = self.children[0].apply(mask=input_mask,
                                         as_list=True, *args,
                                         **kwargs_forw)[0]
        backward = [x[::-1] for x in
                    self.children[1].apply(mask=input_mask,
                                           reverse=True,
                                           as_list=True, *args,
                                           **kwargs_back)][0]
        return [tensor.concatenate([f, b], axis=2)
                for f, b in equizip([forward], [backward])]

    @apply.delegate
    def apply_delegate(self):
        return self.children[0].apply


class GenLSTM(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None, **kwargs):

        super(GenLSTM, self).__init__(**kwargs)
        self.dim = dim
        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation

        self.children = [self.activation, self.gate_activation]

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4
        if name in ['states', 'cells']:
            return self.dim
        if name == 'mask':
            return 0
        return super(GenLSTM, self).get_dim(name)

    def _allocate(self):
        self.W_state = shared_floatx_nans((self.dim, 4*self.dim),
                                          name='W_state')
        self.W_cell_to_in = shared_floatx_nans((self.dim,),
                                               name='W_cell_to_in')
        self.W_cell_to_forget = shared_floatx_nans((self.dim,),
                                                   name='W_cell_to_forget')
        self.W_cell_to_out = shared_floatx_nans((self.dim,),
                                                name='W_cell_to_out')
        # The underscore is required to prevent collision with
        # the `initial_state` application method
        self.initial_state_ = tensor.zeros((self.dim,))
                                                  #name="initial_state")
        self.initial_cells = shared_floatx_zeros((self.dim,),
                                                 name="initial_cells")
        add_role(self.W_state, WEIGHT)
        add_role(self.W_cell_to_in, WEIGHT)
        add_role(self.W_cell_to_forget, WEIGHT)
        add_role(self.W_cell_to_out, WEIGHT)
        add_role(self.initial_state_, INITIAL_STATE)
        add_role(self.initial_cells, INITIAL_STATE)

        self.parameters = [
            self.W_state, self.W_cell_to_in, self.W_cell_to_forget,
            self.W_cell_to_out, self.initial_cells]

    def _initialize(self):
        for weights in self.parameters[:4]:
            self.weights_init.initialize(weights, self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, states, cells, mask=None):
        """Apply the Long Short Term Memory transition.
        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, features). Required for `one_step` usage.
        cells : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current cells in the shape
            (batch_size, features). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            features * 4). The `inputs` needs to be four times the
            dimension of the LSTM brick to insure each four gates receive
            different transformations of the input. See [Grav13]_
            equations 7 to 10 for more details. The `inputs` are then split
            in this order: Input gates, forget gates, cells and output
            gates.
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.
        .. [Grav13] Graves, Alex, *Generating sequences with recurrent
            neural networks*, arXiv preprint arXiv:1308.0850 (2013).
        Returns
        -------
        states : :class:`~tensor.TensorVariable`
            Next states of the network.
        cells : :class:`~tensor.TensorVariable`
            Next cell activations of the network.
        """
        def slice_last(x, no):
            return x[:, no*self.dim: (no+1)*self.dim]
        activation = tensor.dot(states, self.W_state) + inputs
        in_gate = self.gate_activation.apply(
            slice_last(activation, 0) + cells * self.W_cell_to_in)
        forget_gate = self.gate_activation.apply(
            slice_last(activation, 1) + cells * self.W_cell_to_forget)
        next_cells = (
            forget_gate * cells +
            in_gate * self.activation.apply(slice_last(activation, 2)))
        out_gate = self.gate_activation.apply(
            slice_last(activation, 3) + next_cells * self.W_cell_to_out)
        next_states = out_gate * self.activation.apply(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return next_states, next_cells

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.initial_state_[None, :], batch_size, 0),
                tensor.repeat(self.initial_cells[None, :], batch_size, 0)]
