import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import Simple_Node_Embedding


class Similarity_Model(nn.Module):
    def __init__(
        self, original_features_num, num_blocks, in_features, out_features, depth_of_mlp, freeze_mlp
    ):
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
        self.depth_of_mlp = depth_of_mlp
        self.node_embedder = Simple_Node_Embedding(
            original_features_num, num_blocks, in_features, out_features, depth_of_mlp, freeze_mlp
        )

    def forward(self, x):
        x = self.node_embedder(x)
        x = F.normalize(x, dim=2)
        raw_scores = torch.matmul(x, torch.transpose(x, 1, 2))
        return raw_scores
