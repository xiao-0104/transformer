from third_part_library import *

def attention(query, key, value, mask=None, dropout=None):
    d_model = query.size(-1)
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_model)
    if mask is not None:
        scores  =scores.masked.fill(mask==0, 1e-9)
    atten_weight = torch.softmax(scores, dim=-1)
    if dropout is not None:
        atten_weight = dropout(atten_weight)
    return atten_weight * value, atten_weight


class MultiAttention(nn.Module):
    def __init__(self, d_model, head, dropout_p=0.1):
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.each_model = d_model / head
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value, mask=None):

        query, key, value = [linear(x).view(query.size(0), -1, self.head, self.each_model).transpose(1, 2) for linear, x in zip(self.linears, [query, key, value])]
        if mask is not None:
            mask = mask.unsqueeze(1)
        output, atten_weight = attention(query, key, value, mask=mask, dropout=self.dropout)
        output = output.transpose(1, 2).contiguous().view(query.size(0), -1, self.d_model)
        return self.linears[3](output)