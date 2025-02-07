
import time
import copy
import torch

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from divergence_estimation.dense import DenseModule


class Discriminator(nn.Module):
    def __init__(self, layers, dummy_data=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers_ = []
        for i, elem in enumerate(layers):
            ly = DenseModule(elem, activation="leaky_relu", batch_norm=False, dropout=True)
            layers_.append(ly)
        #layers_ += [nn.LazyLinear(1)]

        # Output layer
        self.output_layer = nn.LazyLinear(1)
        layers_.append(self.output_layer)

        self.l = nn.ModuleList(layers_)
        self.loss_plot = None

        if dummy_data is not None:
            # **Forward pass dummy input to initialize LazyLinear layers**
            self._initialize_layers(dummy_data)

            # Initialize weights properly
            self._initialize_weights()

    def _initialize_layers(self, dummy_dl):
        """Perform a dummy forward pass to initialize LazyLinear layers."""
        with torch.no_grad():
            for X, _ in dummy_dl:
                dummy_input = X
                break
            _ = self.forward(dummy_input)


    def _initialize_weights(self):
        """Initialize weights for all layers."""
        for layer in self.l:
            if isinstance(layer, DenseModule):
                nn.init.kaiming_normal_(layer.layer.weight, nonlinearity='leaky_relu')
                if layer.layer.bias is not None:
                    nn.init.constant_(layer.layer.bias, 0.3)

        # Initialize output layer weights
        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='linear')  # For raw score output

        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0.3)

    def forward(self, data: torch.Tensor, *, sigmoid=False) -> torch.Tensor:
        x = data
        for layer in self.l:
            x = layer(x)
        if sigmoid:
            x = F.sigmoid(x)
        return x.reshape(-1)

    def train_loop(self, dataloader: DataLoader, n_epoch: int, optimizer=None, dl_eval=None, n=0, seed=0, cfg=None):
        if optimizer is None:
            if cfg is not None and hasattr(cfg, 'lr'):
                optimizer = optim.Adam(self.parameters(), lr=cfg.lr)
            else:
                optimizer = optim.Adam(self.parameters(), 1e-3)
        self.train(True)
        losses = []
        losses_eval = []
        # Set up early stopping parameters
        best_metric = float('inf')  # For loss, set it to float('inf'); for accuracy, set it to 0
        patience_0 = 1000  # Number of epochs to wait before stopping
        patience = patience_0  # Number of epochs to wait before stopping
        best_model = None

        t_0 = time.time()

        if len(dataloader) == 1: # Only a single batch of data
            X, y = next(iter(dataloader))  # Already get the data, this weirdly goes faster than using the dataloader

        for epoch in (range(n_epoch)):
            if (epoch + 1) % 500 == 0:
                print(f"Discriminator estimator: Epoch [{epoch+1}/{n_epoch}], Time since start: {time.time() - t_0} seconds (average: {(time.time() - t_0) / (epoch+1)} seconds per epoch)")
            cum_loss = 0.0
            cum_loss_eval = 0.0
            self.train(True)
            if len(dataloader) == 1:  # Only a single batch of data
                optimizer.zero_grad()
                logit_X = self(X)
                loss = F.binary_cross_entropy_with_logits(logit_X, y.reshape(-1)) + 1e-3
                loss.backward()
                optimizer.step()
                cum_loss += loss.item()
            else:
                for X, y in dataloader:
                    optimizer.zero_grad()
                    logit_X = self(X)
                    loss = F.binary_cross_entropy_with_logits(logit_X, y.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    cum_loss += loss.item()
            avg_loss = cum_loss / len(dataloader)
            losses.append(avg_loss*2)
            # For early stopping
            if dl_eval is not None:
                self.eval()
                with torch.no_grad():
                    for X_eval, y_eval in dl_eval:
                        logit_X_eval = self(X_eval)
                        loss_eval = F.binary_cross_entropy_with_logits(logit_X_eval, y_eval.reshape(-1))
                        cum_loss_eval += loss_eval.item()

                    avg_loss_eval = cum_loss_eval / (len(dl_eval))
                    losses_eval.append(avg_loss_eval*2)
                # Early stopping
                # Check if the validation loss (or accuracy) has improved
                if avg_loss_eval < best_metric:
                    best_metric = avg_loss_eval
                    patience = patience_0  # Reset patience
                    best_model = copy.deepcopy(self.state_dict())
                else:
                    patience -= 1

                if patience == 0:
                    print("Early stopping, no improvement in validation loss.")
                    self.load_state_dict(best_model)
                    break

    @torch.no_grad()
    def predict(self, data: DataLoader, *, sigmoid=True):
        self.train(False)
        _X_y = 2
        out = []
        for batch in data:
            if len(batch) == _X_y:
                x, _ = batch
            else:
                x = batch
            y = self.forward(x, sigmoid=sigmoid)
            out.append(y)
        return torch.cat(out)

