import torch.nn as nn

class LoRaLin(nn.Module):
    def __init__(self, in_features, out_features, rank_ratio=0.5, bias=True):
        super(LoRaLin, self).__init__()
        rank = max(2,int(min(in_features, out_features) * rank_ratio))
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        return x