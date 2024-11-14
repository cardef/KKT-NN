# src/kkt_network.py

import torch.nn as nn

class KKTNet(nn.Module):
    """
    KKTNet defines a neural network to approximate the solution and Lagrange multipliers
    based on KKT conditions.
    """
    def __init__(self, shared_layers, solution_layers, inequality_multiplier_layers=None, equality_multiplier_layers=None):
        """
        Initializes the KKTNet.

        Args:
            shared_layers (nn.Sequential): Shared layers for feature extraction.
            solution_layers (nn.Sequential): Layers to predict the normalized solution.
            inequality_multiplier_layers (nn.Sequential, optional): Layers to predict inequality multipliers.
            equality_multiplier_layers (nn.Sequential, optional): Layers to predict equality multipliers.
        """
        super(KKTNet, self).__init__()
        self.shared_layers = shared_layers
        self.solution_layers = solution_layers
        self.inequality_multiplier_layers = inequality_multiplier_layers
        self.equality_multiplier_layers = equality_multiplier_layers

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Normalized input variables.

        Returns:
            tuple: Normalized solution and, if present, inequality and equality multipliers.
        """
        shared = self.shared_layers(x)
        solution = self.solution_layers(shared)
        
        ineq_mult = None
        eq_mult = None
        
        if self.inequality_multiplier_layers is not None:
            ineq_mult = self.inequality_multiplier_layers(shared)
        
        if self.equality_multiplier_layers is not None:
            eq_mult = self.equality_multiplier_layers(shared)
        
        return solution, ineq_mult, eq_mult
