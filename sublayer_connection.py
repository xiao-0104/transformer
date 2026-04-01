from third_part_library import *

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout_p=0.1):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

