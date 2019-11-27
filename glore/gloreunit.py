import torch.nn as nn
import torch


class GCN(nn.Module):

    def __init__(self, num_state, num_node):

        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1)

    def forward(self, x):

        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        h = self.conv2(self.relu(h))
        return h


class GloreUnit(nn.Module):

    def __init__(self, num_in, num_nodes):
        super(GloreUnit, self).__init__()

        self.num_out = int(2 * num_nodes)
        self.num_nodes = int(1 * num_nodes)

        # reduce dimensions
        self.conv_reduce = nn.Conv2d(num_in, self.num_out, kernel_size=1)

        # projection map
        self.conv_proj = nn.Conv2d(num_in, self.num_nodes, kernel_size=1)

        self.gcn = GCN(num_state=self.num_out, num_node=self.num_nodes)

        # revert back to original dimensions
        self.conv_revert = nn.Conv2d(self.num_out, num_in, kernel_size=1, bias=False)

        self.batch_norm = nn.BatchNorm2d(num_in, eps=1e-04)

    def forward(self, x):
        """
        :param x: (n, c, h, w)
        """
        n, _, h, w = x.size()

        x_reduced_dim = self.conv_reduce(x).view(n, self.num_out, -1)
        x_projection = self.conv_proj(x).view(n, self.num_nodes, -1)

        x_reverse_projection = x_projection

        # projection: coordinate space -> interaction space
        x_i_space = torch.matmul(x_reduced_dim, x_projection.permute(0, 2, 1))
        x_i_space = x_i_space * (1. / x_reduced_dim.size(2))

        # reasoning with gcn
        x_n_rel = self.gcn(x_i_space)

        # reverse projection: interaction space -> coordinate space
        x_reduced_dim = torch.matmul(x_n_rel, x_reverse_projection)

        x_reduced_reshape = x_reduced_dim.view(n, self.num_out, h, w)
        out = x + self.batch_norm(self.conv_revert(x_reduced_reshape))

        return out

