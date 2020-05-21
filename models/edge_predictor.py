import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel, Simple_Node_Embedding
from models.layers import MlpBlock

class Edge_Predictor(nn.Module):
    def __init__(self, original_features_num, num_blocks, in_features,out_features, depth_of_mlp):
        """
        
        """
        super().__init__()

        self.original_features_num = original_features_num
        self.num_blocks = num_blocks
        self.in_features = in_features
        self.out_features = out_features
        self.depth_of_mlp =depth_of_mlp
        self.edge_embedder = BaseModel(original_features_num, num_blocks, in_features,out_features, depth_of_mlp)
        self.last_block = MlpBlock(out_features,out_features,depth_of_mlp)
        self.last_layer = nn.Conv2d(out_features,1,kernel_size=1, padding=0, bias=True)

    def forward(self, x1):
        x1 = self.edge_embedder(x1)
        x1 = x1.permute(0,3,1,2)
        #print(x1.shape)
        x1 = self.last_block(x1)
        raw_scores1 = self.last_layer(F.relu(x1))#torch.max(x1,3)[0]#[0]
        return torch.sigmoid(raw_scores1.squeeze())#raw_scores1.squeeze()


class Concat_Model(nn.Module):
    def __init__(self, original_features_num, num_blocks, in_features,out_features, depth_of_mlp):
        """
        
        """
        super().__init__()

        self.original_features_num = original_features_num
        self.num_blocks = num_blocks
        self.in_features = in_features
        self.out_features = out_features
        self.depth_of_mlp =depth_of_mlp
        self.node_embedder = Simple_Node_Embedding(original_features_num, num_blocks, in_features,out_features, depth_of_mlp)
        #self.last_block = MlpBlock(2*320,out_features,depth_of_mlp)
        self.last_layer = nn.Conv2d(640,1,kernel_size=1, padding=0, bias=True)

    def forward(self, x1):
        x1 = self.node_embedder(x1)
        n_vertices = x1.shape[1]
        feature = x1.shape[-1]
        x11 = x1.unsqueeze(1)
        x11 = x11.expand(-1,n_vertices,-1,-1)
        x12 = x1.unsqueeze(2)
        x12 = x12.expand(-1,-1,n_vertices,-1)
        #print(x11.shape, x12.shape)
        x1 = torch.cat((x11,x12),3)
        #print(x1.shape)
        x1 = x1.permute(0,3,1,2)
        #x1 = self.last_block(x1)
        raw_scores1 = self.last_layer(x1)
        #raw_scores = torch.matmul(x1,torch.transpose(x1, 1, 2))
        return F.sigmoid(raw_scores1).squeeze()#raw_scores1.squeeze()