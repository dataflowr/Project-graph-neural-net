import torch

def adjacency_matrix_to_tensor_representation(W, with_degree= True):
    """ Create a tensor B[:,:,1] = W and B[i,i,0] = deg(i)"""
    degrees = W.sum(1)
    B = torch.zeros((len(W), len(W), 2))
    B[:, :, 1] = W
    indices = torch.arange(len(W))
    if with_degree:
        B[indices, indices, 0] = degrees
    return B