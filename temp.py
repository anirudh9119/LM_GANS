import theano
import theano.tensor as tensor

output_mask = tensor.vector('output',  dtype='int32')
a = tensor.vector('a', dtype='int32')
b = tensor.vector('b', dtype='int32')
c= tensor.vector('b', dtype='int32')

d = tensor.cumsum(c)
emb = tensor.switch(tensor.eq(output_mask, 0), d-1,d-1)


f = theano.function([output_mask, a , b,c] , [emb], on_unused_input='warn')

print f([1,0,1,0],[1,2,3,4],[5,6,7,8], [1,1,1,1])

