import numpy as np

'''
indices is a matrix of integers.  

sequence x examples

Map to: 

sequence x examples x 30k
'''

def to_one_hot(indices):
    num_labels = 30000
    xlst = []
    for i in range(0, indices.shape[0]):
        xlst.append(np.asarray([np.eye(30000)[indices[i]]]))


    return np.vstack(xlst).transpose(1,0,2).astype('float32')

if __name__ == "__main__":



    indices = np.asarray([[1,2,3],[4,5,6]]).astype('int32')

    print to_one_hot(indices).dtype

