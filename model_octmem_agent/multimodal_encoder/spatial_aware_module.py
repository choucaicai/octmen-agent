from torch import nn
from .position_encodings import PositionEmbeddingLearnedMLP


class SpatialAwareModule(nn.Module):

    def __init__(self,dim=3, latent_dim=1024):
        super(SpatialAwareModule, self).__init__()
        self.dim = dim
        self.positional_embedding = PositionEmbeddingLearnedMLP(dim=dim, num_pos_feats=latent_dim)

    def encode_pe(self, xyz=None):
        return self.positional_embedding(xyz)
