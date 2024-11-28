
import torch.nn as nn
class ResidualBlock(nn.Module):
    """
    Defines a residual block used in the neural network.
    """

    def __init__(self, n):
        """
        Initializes the ResidualBlock.

        Args:
            n (int): Number of input and output features.
        """
        super().__init__()
        self.linear = nn.Linear(n, n)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the residual block.
        """
        identity = x
        out = self.linear(x)
        out = self.relu(out)
        return self.relu(out + identity)


class Net(nn.Module):
    """
    Neural network to learn optimal solutions based on KKT conditions.
    """

    def __init__(
        self,
        input_dim,
        num_eq_constraints,
        num_ineq_constraints,
        num_decision_vars,
        hidden_dim=512,
        num_residual_blocks=4,
    ):
        """
        Initializes the net.

        Args:
            input_dim (int): Number of parameters.
            num_eq_constraints (int): Number of equality constraints.
            num_ineq_constraints (int): Number of inequality constraints.
            num_decision_vars (int): Number of decision variables.
            hidden_dim (int, optional): Hidden layer size. Defaults to 512.
            num_residual_blocks (int, optional): Number of residual blocks. Defaults to 4.
        """
        super(Net, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU()]
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.shared = nn.Sequential(*layers)

        # Output for decision variables
        self.decision_output = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, num_decision_vars),
            nn.Tanh(),  # Outputs between -1 and 1
        )

        # Conditionally create outputs for dual variables
        if num_eq_constraints > 0:
            self.dual_eq_output = nn.Sequential(
                ResidualBlock(hidden_dim),
                ResidualBlock(hidden_dim),
                nn.Linear(hidden_dim, num_eq_constraints),
            )
        else:
            self.dual_eq_output = None  # No equality constraints

        if num_ineq_constraints > 0:
            self.dual_ineq_output = nn.Sequential(
                ResidualBlock(hidden_dim),
                ResidualBlock(hidden_dim),
                nn.Linear(hidden_dim, num_ineq_constraints),
                nn.Softplus(beta=5),  # Ensures outputs are non-negative
            )
        else:
            self.dual_ineq_output = None  # No inequality constraints

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            tuple: Outputs for decision variables, dual equality variables, and dual inequality variables.
        """
        embedding = self.shared(x)
        decision = self.decision_output(embedding)
        dual_eq = (
            self.dual_eq_output(embedding) if self.dual_eq_output is not None else None
        )
        dual_ineq = (
            self.dual_ineq_output(embedding)
            if self.dual_ineq_output is not None
            else None
        )
        return decision, dual_eq, dual_ineq
