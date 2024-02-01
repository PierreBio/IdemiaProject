from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, layers):
        """
        Initializes a Multi-Layer Perceptron with a specified number of layers,
        using ReLU as the activation function.

        Args:
            input_size (int): The size of the input features.
            output_size (int): The size of the output.
            layers (list of int): A list where each element is the size of a hidden layer.
        """
        super(MLP, self).__init__()
        self.m_layers = nn.Sequential()

        # Dynamically create hidden layers with ReLU activation
        for i, layer_size in enumerate(layers, start=1):
            self.m_layers.add_module(f"Linear_{i}", nn.Linear(input_size, layer_size))
            self.m_layers.add_module(f"Activation_{i}", nn.ReLU())
            input_size = layer_size

        self.m_layers.add_module("Output", nn.Linear(input_size, output_size))

    def forward(self, x):
        return self.m_layers(x)
