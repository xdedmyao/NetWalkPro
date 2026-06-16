import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, dimensions, activation=torch.sigmoid, lamb=0.01, gamma=0.01, beta=0.01, rho=0.4, seed=24):
        """
        dimensions: list, e.g. [input_dim, hidden1, hidden2, ..., latent_dim]
        activation: function, e.g. torch.sigmoid or torch.relu
        lamb, gamma, beta, rho: hyperparams (same meaning as TF 源代码)
        """
        super(Autoencoder, self).__init__()
        torch.manual_seed(seed)
        self.dimensions = dimensions
        self.n_layers = len(dimensions) - 1
        self.activation = activation
        self.lamb = lamb
        self.gamma = gamma
        self.beta = beta
        self.rho = rho

        self.W_enc = nn.ParameterList()
        self.b_enc = nn.ParameterList()
        for i in range(self.n_layers):
            n_input = dimensions[i]
            n_output = dimensions[i + 1]
            W = nn.Parameter(torch.empty((n_output, n_input)))
            nn.init.uniform_(W, a=-1.0 / np.sqrt(n_input), b=1.0 / np.sqrt(n_input))
            b = nn.Parameter(torch.zeros((1, n_output)))
            self.W_enc.append(W)
            self.b_enc.append(b)

        self.W_dec = nn.ParameterList()
        self.b_dec = nn.ParameterList()

        for i in range(self.n_layers - 1, -1, -1):
            n_input = dimensions[i + 1]
            n_output = dimensions[i]
            W = nn.Parameter(torch.empty((n_output, n_input)))
            nn.init.uniform_(W, a=-1.0 / np.sqrt(n_input), b=1.0 / np.sqrt(n_input))
            b = nn.Parameter(torch.zeros((1, n_output)))
            self.W_dec.append(W)
            self.b_dec.append(b)

    def forward(self, x, corrupt_prob=0.0):
        """
        x: torch.Tensor shape [input_dim, batch_size] (float)
        corrupt_prob: float (0..1) or torch scalar
        返回：
          encoder_out [latent_dim, batch_size]
          reconstruction [input_dim, batch_size]
          noise_input [input_dim, batch_size]
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        noise = torch.rand_like(x) * 0.1
        r = x + noise
        # current_input = corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)
        cp = float(corrupt_prob) if not torch.is_tensor(corrupt_prob) else float(corrupt_prob.item())
        current_input = r * cp + x * (1.0 - cp)
        noise_input = current_input

        cur = current_input
        weight_decay_J = 0.0
        for i in range(self.n_layers):
            W = self.W_enc[i]
            b = self.b_enc[i]  # shape [1, n_output]
            # W @ cur -> [n_output, batch_size], add bias (broadcast on columns)
            out = self.activation(torch.matmul(W, cur) + b.t().transpose(0, 1)) if False else None
            out = self.activation(torch.matmul(W, cur) + b.t())
            cur = out
            weight_decay_J = weight_decay_J + (self.lamb / 2.0) * torch.mean(W ** 2)

        encoder_out = cur  # shape [latent_dim, batch_size]

        cur_dec = encoder_out
        for i in range(len(self.W_dec)):
            W = self.W_dec[i]
            b = self.b_dec[i]
            cur_dec = self.activation(torch.matmul(W, cur_dec) + b.t())

            weight_decay_J = weight_decay_J + (self.lamb / 2.0) * torch.mean(W ** 2)

        reconstruction = cur_dec  # shape [input_dim, batch_size]

        # encoder_out: [latent_dim, batch_size] -> transpose -> [batch_size, latent_dim], reduce_mean over batch -> [latent_dim]
        rhohats = torch.mean(encoder_out.t(), dim=0)  # shape [latent_dim]
        rho = self.rho
        eps = 1e-10
        kl = torch.mean(rho * torch.log((rho + eps) / (rhohats + eps)) + (1 - rho) * torch.log(((1 - rho) + eps) / ((1 - rhohats) + eps)))

        ae_loss = (self.gamma / 2.0) * torch.mean((reconstruction - x) ** 2)
        kl_loss = self.beta * kl
        cost = ae_loss + kl_loss + weight_decay_J

        return {
            'x': x,
            'encoder_out': encoder_out,
            'reconstruction': reconstruction,
            'corrupt_prob': corrupt_prob,
            'cost': cost,
            'noise_input': noise_input,
            'kl': kl,
            'weight_decay_J': weight_decay_J,
            'ae_loss': ae_loss,
            'kl_loss': kl_loss,
            'W_list': list(self.W_enc),
            'b_list': list(self.b_enc)
        }


def autoencoder(data, corrupt_prob, dimensions, beta=0.01, rho=0.4, activation=torch.sigmoid, lamb=0.01, gamma=0.01):
    ae = Autoencoder(dimensions, activation=activation, lamb=lamb, gamma=gamma, beta=beta, rho=rho)
    if data is None:
        return {
            'module': ae,
            'W_list': list(ae.W_enc),
            'b_list': list(ae.b_enc),
            'x': None,
            'encoder_out': None,
            'reconstruction': None,
            'corrupt_prob': corrupt_prob,
            'cost': None,
            'noise_input': None,
            'kl': None,
            'weight_decay_J': None,
            'ae_loss': None,
            'kl_loss': None
        }
    else:
        if isinstance(data, np.ndarray):
            x = torch.tensor(data, dtype=torch.float32)
        else:
            x = data
        return ae.forward(x, corrupt_prob)
