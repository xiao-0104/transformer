from third_part_library import *

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

