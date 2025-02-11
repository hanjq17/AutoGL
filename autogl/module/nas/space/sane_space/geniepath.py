from torch_geometric.nn import GATConv
import torch


class Breadth(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Breadth, self).__init__()
        self.gatconv = GATConv(in_dim, out_dim, heads=1)

    def forward(self, x, edge_index, size=None):
        x = torch.tanh(self.gatconv(x, edge_index, size=size))
        return x


class Depth(torch.nn.Module):

    def __init__(self, in_dim, hidden):
        super(Depth, self).__init__()
        self.lstm = torch.nn.LSTM(in_dim, hidden, 1, bias=False)

    def forward(self, x, h, c):
        x, (h, c) = self.lstm(x, (h, c))
        return x, (h, c)


class GeniePathLayer(torch.nn.Module):

    def __init__(self, in_dim, hidden):
        super(GeniePathLayer, self).__init__()
        self.breadth_func = Breadth(in_dim, hidden)
        self.depth_func = Depth(hidden, hidden)
        self.in_dim = in_dim
        self.hidden = hidden
        self.lstm_hidden = hidden

    def forward(self, x, edge_index, size=None):
        if torch.is_tensor(x):
            h = torch.zeros(1, x.shape[0], self.lstm_hidden, device=x.device)
            c = torch.zeros(1, x.shape[0], self.lstm_hidden, device=x.device)
        else:
            h = torch.zeros(1, x[1].shape[0], self.lstm_hidden, device=x[1].device)
            c = torch.zeros(1, x[1].shape[0], self.lstm_hidden, device=x[1].device)
        x = self.breadth_func(x, edge_index, size=size)
        x = x[None, :]
        x, (h, c) = self.depth_func(x, h, c)
        x = x[0]
        return x