from torch import nn

_activations = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(0.2),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
}


class DenseModule(nn.Module):
    def __init__(self, n_neurons: int, activation: str, *args, batch_norm: bool, dropout: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = nn.LazyLinear(out_features=n_neurons)
        if activation not in _activations.keys():
            msg = f"Expected one of {_activations.keys()}"
            raise ValueError(msg)
        self.activation = _activations[activation]
        self.batch_norm = None
        if batch_norm:
            self.batch_norm = nn.LazyBatchNorm1d()
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.layer(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x