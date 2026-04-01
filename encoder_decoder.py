from third_part_library import *
from feed_forward import *
from input import *
from multi_attention import *
from sublayer_connection import *

def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, d_model, MultiAttention, FeedForward, dropout_p):
        super().__init__()
        self.d_model = d_model
        self.MultiAttention = MultiAttention
        self.FeedForward = FeedForward
        self.sublayers = clone(SublayerConnection(d_model, dropout_p), 2)

    def forward(self, x, mask=None):
        x = self.sublayers[0](x, lambda x: self.MultiAttention(x, x, x, mask))
        return self.sublayers[1](x, FeedForward)


class Decoder(nn.Module):
    def __init__(self, d_model, MultiAttention, FeedForward, dropout_p):
        super().__init__()
        self.d_model = d_model
        self.MultiAttention = MultiAttention
        self.FeedForward = FeedForward
        self.sublayers = clone(SublayerConnection(d_model, dropout_p), 3)

    def forward(self, x, memory, source_mask, target_mask):
        x = self.sublayers[0](x, lambda x: self.MultiAttention(x, x, x, source_mask))
        x = self.sublayers[1](x, lambda x: self.MultiAttention(x, memory, memory, target_mask))
        return self.sublayers[2](x, FeedForward)