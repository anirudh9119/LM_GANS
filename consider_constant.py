import theano
import theano.tensor as T
from theano.tensor.opt import register_canonicalize

class ConsiderConstant(theano.compile.ViewOp):
    def grad(self, args, g_outs):
        return [T.zeros_like(g_out) for g_out in g_outs]

consider_constant = ConsiderConstant()
register_canonicalize(theano.gof.OpRemove(consider_constant), name='r_consider_constant')

