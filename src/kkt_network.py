# kkt_network.py
import torch
from torch import nn

class KKTNet(nn.Module):
    def __init__(self, shared_net, solution_net, inequality_multiplier_net=None, equality_multiplier_net=None):
        super().__init__()
        self.shared = shared_net
        self.solution = solution_net
        self.inequality_multiplier_net = inequality_multiplier_net
        self.equality_multiplier_net = equality_multiplier_net

    def forward(self, inputs):
        embedding = self.shared(inputs)
        solution = self.solution(embedding)
        outputs = [solution]

        if self.inequality_multiplier_net is not None:
            inequality_multipliers = self.inequality_multiplier_net(embedding)
            outputs.append(inequality_multipliers)
        else:
            outputs.append(None)

        if self.equality_multiplier_net is not None:
            equality_multipliers = self.equality_multiplier_net(embedding)
            outputs.append(equality_multipliers)
        else:
            outputs.append(None)

        return outputs
