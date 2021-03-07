import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class triplet_loss(nn.Module):
    def __init__(
        self, device="cpu", loss_reduction="mean", loss=nn.CrossEntropyLoss(reduction="sum")
    ):
        super(triplet_loss, self).__init__()
        self.device = device
        self.loss = loss
        if loss_reduction == "mean":
            self.increments = lambda new_loss, n_vertices: (new_loss, n_vertices)
        elif loss_reduction == "mean_of_mean":
            self.increments = lambda new_loss, n_vertices: (new_loss / n_vertices, 1)
        else:
            raise ValueError("Unknown loss_reduction parameters {}".format(loss_reduction))

    def forward(self, outputs):
        """
        outputs is the output of siamese network (bs,n_vertices,n_vertices)
        """
        loss = 0
        total = 0
        for out in outputs:
            n_vertices = out.shape[0]
            ide = torch.arange(n_vertices)
            target = ide.to(self.device)
            incrs = self.increments(self.loss(out, target), n_vertices)
            loss += incrs[0]
            total += incrs[1]
        return loss / total


def get_criterion(device, loss_reduction):
    return triplet_loss(device, loss_reduction)


# TODO refactor


class tsp_loss(nn.Module):
    def __init__(self, loss=nn.BCELoss(reduction="none")):
        super(tsp_loss, self).__init__()
        self.loss = loss
        self.normalize = torch.nn.Sigmoid()  # Softmax(dim=2)

    def forward(self, raw_scores, mask, target):
        """
        raw_scores (bs,n_vertices,n_vertices)
        """
        proba = self.normalize(raw_scores)
        return torch.mean(mask * self.loss(proba, target))


class cluster_similarity_loss(nn.Module):
    def __init__(self, loss=nn.MSELoss(reduction="mean")):
        super(cluster_similarity_loss, self).__init__()
        self.loss = loss

    def forward(self, raw_scores, cluster_sizes):
        """
        raw_scores (bs,n_vertices,n_vertices)
        cluster_sizes (bs, n_clusters)
        """
        target = torch.zeros_like(raw_scores)
        for i, n_nodes in enumerate(cluster_sizes):
            prev = 0
            for n in n_nodes:
                target[i][prev : prev + n, prev : prev + n] = torch.ones(n, n)
                prev = n
        return self.loss(raw_scores, target)


class cluster_embedding_loss(nn.Module):
    def __init__(self, device="cpu"):
        super(cluster_embedding_loss, self).__init__()
        self.device = device

    def forward(self, embeddings, cluster_sizes):
        """
        embeddings (bs,n_vertices,dim_embedding)
        cluster_sizes (bs, n_clusters)
        """
        loss = torch.zeros([1], dtype=torch.float64, device=self.device)
        for i, n_nodes in enumerate(cluster_sizes):
            mean_cluster = []
            var_cluster = []
            prev = 0
            for n in n_nodes:
                mean_cluster.append(
                    F.normalize(torch.mean(embeddings[i][prev : prev + n, :], 0), dim=0)
                )
                var_cluster.append(torch.var(embeddings[i][prev : prev + n, :], 0))
                prev = n

            loss_dist_cluster = torch.zeros([1], dtype=torch.float64, device=self.device)
            loss_var_cluster = torch.zeros([1], dtype=torch.float64, device=self.device)
            for m1, m2 in itertools.combinations(mean_cluster, 2):
                loss_dist_cluster += 1.0 + torch.dot(m1, m2)

            mu = 1.0
            loss_var_cluster = mu * torch.stack(var_cluster, dim=0).sum()

            loss += 0.1 * loss_dist_cluster + loss_var_cluster
        return loss
