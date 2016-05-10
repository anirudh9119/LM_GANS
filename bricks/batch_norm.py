import logging
import numpy

from blocks.bricks import Activation
from blocks.bricks.base import application, lazy
from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import MonitoringExtension
from blocks.filter import get_brick
from blocks.graph import ComputationGraph
from blocks.roles import add_role, PARAMETER
from blocks.utils import dict_subset

from theano import config, shared, tensor, function
import theano.tensor.nnet.bn as bn

floatX = config.floatX
logger = logging.getLogger()

class BatchNorm(Activation):
    """Brick for Batch Normalization. It works with 4D Tensors (conv.) and
    2D Tensors (fully connected layers).
    The Batch Normalization paper:
    S. Ioffe, C. Szegedy, Batch Normalization: Accelerating Deep Network
    Training by Reducing Internal Covariate Shift.
    Parameters
    ----------
    input_dim : int
        The number of features (or features maps for convolutions).
    epsilon : float
        Small constant for variance stability.
    Examples
    --------
    >>> import theano
    >>> from theano import tensor
    >>> x = tensor.vector('x')
    Creating a network:
    >>> y = Linear(input_dim=10, output_dim=5).apply(x)
    >>> bn = BatchNorm(input_dim=5)
    >>> train_out = bn.apply(y)
    Creating both train and test computation graphs:
    >>> train_cg = ComputationGraph([train_out])
    >>> test_cg = create_inference_graph(train_cg, [bn])
    Preparing the update extension:
    >>> batch_size = 50 #The size of the batches
    >>> n_batches = 10 #The number of batches to use to update the stats.
    >>> scheme = ShuffledScheme(batch_size*n_batches, batch_size)
    >>> stream = DataStream(DATASET, iteration_scheme=scheme)
    >>> extensions.insert(0, BatchNormExtension([bn], stream, n_batches))
    """
    @lazy(allocation=['input_dim'])
    def __init__(self, input_dim, epsilon=1e-6, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.epsilon = epsilon

    @property
    def gamma(self):
        return self.parameters[0]

    @property
    def beta(self):
        return self.parameters[1]

    def _allocate(self):
        gamma_val = numpy.ones(self.input_dim, dtype=floatX)
        gamma = shared(name='gamma', value=gamma_val)
        beta_val = numpy.zeros(self.input_dim, dtype=floatX)
        beta = shared(name='beta', value=beta_val)
        add_role(gamma, PARAMETER)
        add_role(beta, PARAMETER)
        self.parameters.append(gamma)
        self.parameters.append(beta)
        # Keeping track of the means and variances during the training.
        means_val = numpy.zeros(self.input_dim, dtype=floatX)
        self.pop_means = shared(name='means', value=means_val)
        vars_val = numpy.ones(self.input_dim, dtype=floatX)
        self.pop_vars = shared(name='varainces', value=vars_val)

    def get_replacements(self):
        """Returns the replacements for the computation graph."""
        return {self.training_output: self.inference_output}

    def get_updates(self, n_batches):
        """Update the population means and variances of the brick. Use
        n_batches from the training dataset to do so.
        """
        m_u = (self.pop_means, (self.pop_means +
                                1./n_batches * self.batch_means))
        v_u = (self.pop_vars, (self.pop_vars +
                               1./n_batches * self.batch_vars))
        return [m_u, v_u]

    def _inference(self, input_):
        output = bn.batch_normalization(input_,
            self.gamma.dimshuffle(*self.pattern),
            self.beta.dimshuffle(*self.pattern),
            self.pop_means.dimshuffle(*self.pattern),
            tensor.sqrt(self.pop_vars.dimshuffle(*self.pattern) +
                        self.epsilon),
            mode='low_mem')
        return output

    def _training(self, input_):
        self.batch_means = input_.mean(axis=self.axes, keepdims=False,
                                       dtype=floatX)
        self.batch_vars = input_.var(axis=self.axes, keepdims=False)
        output = bn.batch_normalization(input_,
            self.gamma.dimshuffle(*self.pattern),
            self.beta.dimshuffle(*self.pattern),
            self.batch_means.dimshuffle(*self.pattern),
            tensor.sqrt(self.batch_vars.dimshuffle(*self.pattern) +
                        self.epsilon),
            mode='low_mem')
        return output

    @application
    def apply(self, input_):
        if input_.ndim == 2:
            self.axes = [0]
            self.pattern = ['x', 0]
        elif input_.ndim == 4:
            self.axes = [0, 2, 3]
            self.pattern = ['x', 0, 'x', 'x']
        elif input_.ndim == 3:
            self.axes = [0, 1]
            self.pattern = ['x', 'x', 0]
        else:
            raise NotImplementedError
        self.training_output = self._training(input_)
        self.inference_output = self._inference(input_)
        return self.training_output


class MaskedBatchNorm(BatchNorm):
    @lazy(allocation=['input_dim'])
    def __init__(self, input_dim, epsilon=1e-6, **kwargs):
        super(MaskedBatchNorm, self).__init__(input_dim=input_dim,
                                              epsilon=epsilon,
                                              **kwargs)
    def _inference(self, input_, input_mask):
        input_ = input_ * input_mask.dimshuffle(*self.mask_pattern)
        output = input_ - self.pop_means.dimshuffle(*self.pattern)
        output /= tensor.sqrt(self.pop_vars.dimshuffle(*self.pattern)
                              + self.epsilon)
        output *= self.gamma.dimshuffle(*self.pattern)
        output += self.beta.dimshuffle(*self.pattern)
        return output * input_mask.dimshuffle(*self.mask_pattern)

    def _training(self, input_, input_mask):
        n = (input_mask.sum(dtype=floatX,
                            keepdims=False) * self.shape).astype(floatX)
        input_ = input_ * input_mask.dimshuffle(*self.mask_pattern)
        self.batch_means = input_.sum(axis=self.axes, keepdims=False,
                                      dtype=floatX)
        self.batch_means /= n
        self.batch_vars = (input_**2).sum(axis=self.axes, keepdims=False,
                                          dtype=floatX)
        self.batch_vars /= n
        self.batch_vars -= self.batch_means**2
        output = input_ - self.batch_means.dimshuffle(*self.pattern)
        output /= tensor.sqrt(self.batch_vars.dimshuffle(*self.pattern)
                              + self.epsilon)
        output *= self.gamma.dimshuffle(*self.pattern)
        output += self.beta.dimshuffle(*self.pattern)
        return output * input_mask.dimshuffle(self.mask_pattern)

    @application(inputs=['input_', 'input_mask'], outputs=['output'])
    def apply(self, input_, input_mask=None):
        if input_.ndim == 4:
            self.axes = [0, 2, 3]
            self.pattern = ['x', 0, 'x', 'x']
            self.shape = input_.shape[2]
            self.mask_pattern = [1, 'x', 'x', 0]
        elif input_.ndim == 3:
            self.axes = [0, 1]
            self.pattern = ['x', 'x', 0]
            self.shape = 1
            self.mask_pattern = [1, 0, 'x']
        else:
            raise NotImplementedError
        self.training_output = self._training(input_, input_mask)
        self.inference_output = self._inference(input_, input_mask)
        return self.training_output

class BatchNormExtension(SimpleExtension, MonitoringExtension):
    """Computes the population means and variance of the BatchNorm bricks
    in the network. This extension must be placed before any other
    monitoring.
    Parameters
    ----------
    graph : instance of :class:`ComputationGraph`
        The training computation graph.
    data_stream : instance of :class:`DataStream`
        The data stream used to compute the population statistics on. It
        should provide n_batches only.
    n_batches: int
        The number of batches used to update the population statistics.
    """
    def __init__(self, graph, data_stream, n_batches, **kwargs):
        kwargs.setdefault("after_epoch", True)
        kwargs.setdefault("before_first_epoch", True)
        super(BatchNormExtension, self).__init__(**kwargs)
        self.n_batches = n_batches
        self.bricks = get_batch_norm_bricks(graph)
        self.data_stream = data_stream
        self.updates = self._get_updates()
        variables = [brick.training_output for brick in self.bricks]
        self._computation_graph = ComputationGraph(variables)
        self.inputs = self._computation_graph.inputs
        self.inputs = list(set(self.inputs))
        self.inputs_names = [v.name for v in self.inputs]
        self._compile()

    def _get_updates(self):
        updates = []
        for brick in self.bricks:
            updates.extend(brick.get_updates(self.n_batches))
        return updates

    def _reset(self, x):
        x.set_value(numpy.zeros(x.get_value().shape, dtype=floatX))

    def _compile(self):
        self._fun = function(self.inputs, [], updates=self.updates,
                             on_unused_input='ignore')
    def _evaluate(self):
        for batch in self.data_stream.get_epoch_iterator(as_dict=True):
            batch = dict_subset(batch, self.inputs_names)
            self._fun(**batch)


    def do(self, which_callback, *args):
        logger.info('Computation of population statistics started')
        # 1. Reset the pop means and vars
        for brick in self.bricks:
            self._reset(brick.pop_means)
            self._reset(brick.pop_vars)
        # 2. Update them
        self._evaluate()
        logger.info('Computation of population statistics finished')


def create_inference_graph(graph):
    """Create the inference graph from the training computation graph.
    Parameters
    ----------
    graph : instance of :class:`ComputationGraph`
        The training computation graph.
    """
    replacements = {}
    bricks = get_batch_norm_bricks(graph)
    for brick in bricks:
        replacements.update(brick.get_replacements())
    return graph.replace(replacements)


def get_batch_norm_bricks(graph):
    """Returns the batch norm bricks (BatchNorm and BatchNorm3D) in a
       computation graph.
    Parameters
    ----------
    graph : instance of :class:`ComputationGraph`
        The training computation graph.
    """
    bricks = []
    for variable in graph.variables:
        brick = get_brick(variable)
        if isinstance(brick, MaskedBatchNorm):
            if brick not in bricks:
                bricks.append(brick)
    return bricks
