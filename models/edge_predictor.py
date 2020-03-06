import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel

class Edge_Predictor(nn.Module):
    def __init__(self, original_features_num, num_blocks, in_features,out_features, depth_of_mlp):
        """
        take a batch of pair of graphs 
        ((bs, n_vertices, n_vertices, in_features) (bs,n_vertices, n_vertices, in_features))
        and return a batch of node similarities (bs, n_vertices, n_vertices)
        for each node the sum over the second dim should be one: sum(torch.exp(out[b,i,:]))==1
        graphs must have same size inside the batch
        """
        super().__init__()

        self.original_features_num = original_features_num
        self.num_blocks = num_blocks
        self.in_features = in_features
        self.out_features = out_features
        self.depth_of_mlp =depth_of_mlp
        self.edge_embedder = BaseModel(original_features_num, num_blocks, in_features,out_features, depth_of_mlp)

    def forward(self, x1):
        x1 = self.edge_embedder(x1)
        raw_scores = torch.max(x1,3)[0]
        return F.log_softmax(raw_scores, dim = 2)
