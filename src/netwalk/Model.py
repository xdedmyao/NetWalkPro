import numpy as np
import torch
import torch.optim as optim
from scipy.sparse import csgraph
from tqdm import tqdm

from netwalk.DenoisingAE import autoencoder



class Model:
    def __init__(self, activation, dimension, walk_len, nodeNum, gama, lamb,
                 beta, rho, epoch, batch_size, learning_rate, optimizer_type, corrupt_prob, device=None):

        self.activation = activation
        self.corrupt_prob_value = corrupt_prob
        self.optimized = False
        self.dimension = dimension
        self.walk_len = walk_len
        self.gama = gama
        self.lamb = lamb
        self.beta = beta
        self.rho = rho
        self.epoch = epoch
        self.batch_size = batch_size * self.walk_len
        self.learning_rate = learning_rate
        self.nodeNum = nodeNum
        self.optimizer_type = optimizer_type

        # Device selection: CUDA > MPS (Apple Silicon) > CPU
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # use the following code if you do not have a cuda gpu
        # self.device = device if device is not None else ("cpu")

        ae_info = autoencoder(None, self.corrupt_prob_value, self.dimension, beta=self.beta, rho=self.rho,
                              activation=self.activation, lamb=self.lamb, gamma=self.gama)

        self.ae = ae_info['module'].to(self.device)


        # self.ae = torch.compile(self.ae, mode="reduce-overhead")

        self.W_list = ae_info['W_list']
        self.b_list = ae_info['b_list']

        self.data = None

        self.encoder_out = None

        params = list(self.ae.parameters())
        if self.optimizer_type == "adam":
            self.optimizer = optim.Adam(params, lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        elif self.optimizer_type == "adagrad":
            self.optimizer = optim.Adagrad(params, lr=self.learning_rate, eps=1e-8)
        elif self.optimizer_type == "gd":
            self.optimizer = optim.SGD(params, lr=self.learning_rate)
        elif self.optimizer_type == "rmsprop":
            self.optimizer = optim.RMSprop(params, lr=self.learning_rate)
        elif self.optimizer_type == "momentum":
            self.optimizer = optim.SGD(params, lr=self.learning_rate, momentum=0.95)
        elif self.optimizer_type == "lbfgs":
            self.optimizer = optim.LBFGS(params, lr=self.learning_rate)
        else:
            self.optimizer = optim.Adam(params, lr=self.learning_rate)

        self.loss = None
        self.clique_loss = None
        self.ae_loss = None
        self.kl_loss = None
        self.self_weight_decay_J = None

        phi = np.ones((self.walk_len, self.walk_len)) - np.eye(self.walk_len)
        L_np = csgraph.laplacian(phi, normed=False)

        self.L = torch.tensor(
            L_np,
            dtype=torch.float32,
            device=self.device
        )  # shape: [walk_len, walk_len]

    def feedforward_autoencoder(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        current_input = data.astype(np.float32)
        x = torch.tensor(current_input, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            res_dict = self.ae.forward(x, corrupt_prob=self.corrupt_prob_value)
            encoder_out = res_dict['encoder_out']  # shape [latent_dim, batch]

            out = encoder_out.t().cpu().numpy()
            return np.array(out)

    def sigmoid(self, x, derivative=False):
        sigm = 1. / (1. + np.exp(-x))
        if derivative:
            return sigm * (1. - sigm)
        return sigm

    def _compute_losses(self, batch_tensor):

        out = self.ae.forward(batch_tensor, corrupt_prob=self.corrupt_prob_value)
        encoder_out = out['encoder_out']  # [latent_dim, batch_size]
        reconstruction = out['reconstruction']
        ae_loss = out['ae_loss']
        ae_cost = out['cost']
        kl = out['kl']
        weight_decay_J = out['weight_decay_J']

        trans_code = encoder_out.t()
        # reshape -> [-1, walk_len, latent_dim]
        batch_size = trans_code.shape[0]
        latent_dim = trans_code.shape[1]

        if batch_size % self.walk_len != 0:
            num_walks = batch_size // self.walk_len
            trans_code = trans_code[:num_walks * self.walk_len, :]
        else:
            num_walks = batch_size // self.walk_len

        if num_walks == 0:
            clique_J = torch.tensor(0.0, device=self.device)
        else:
            trans_code = trans_code.view(num_walks, self.walk_len, latent_dim)  # [num_walks, walk_len, latent_dim]
            t_trans_code = trans_code.permute(0, 2, 1)  # [num_walks, latent_dim, walk_len]

            clique_J = torch.einsum('aij,jk,aki->a', t_trans_code, self.L, trans_code).mean()


        clique_loss = clique_J
        loss = clique_loss + ae_cost
        return loss, clique_loss, ae_loss, kl, weight_decay_J, encoder_out

    def clique_embedding_loss(self):
        self.optimized = True
        return None, None, None, None, None

    def print_loss(self, loss_evaled, cl, ae, kl, weight_loss):
        print(loss_evaled, " cl:", cl, " ae:", ae, " kl:", kl, " l2_regularizer:", weight_loss)

    def batchify(self, data, bsz, shuffle=False):
        # data: torch.Tensor on GPU
        if shuffle:
            perm = torch.randperm(data.shape[1], device=data.device)
            data = data[:, perm]

        nbatch = data.shape[1] // bsz
        batches = []
        for i in range(nbatch):
            batch = data[:, i * bsz:(i + 1) * bsz]
            batches.append(batch)
        return batches

    def fit(self, data_train):
        if isinstance(data_train, np.ndarray):
            data_train = torch.tensor(
                data_train,
                dtype=torch.float32,
                device=self.device
            )
        else:
            data_train = data_train.to(self.device)

        epochs = range(1, self.epoch + 1)
        for epoch in tqdm(epochs):
            batches = self.batchify(data_train, self.batch_size)
            bt_loss = 0.0
            if self.optimizer_type == "lbfgs":
                for batch in batches:
                    batch_tensor = batch
                    def closure():
                        self.optimizer.zero_grad()
                        loss, clique_loss, ae_loss, kl, weight_decay_J, encoder_out = self._compute_losses(batch_tensor)
                        loss.backward()
                        return loss
                    self.optimizer.step(closure)
                    # compute values to print
                    loss_evaled, clique_loss_val, ae_val, kl_val, weight_val, encoder_out = self._compute_losses(batch_tensor)
                    bt_loss += float(loss_evaled.item())
            else:
                for batch in batches:
                    batch_tensor = batch
                    self.optimizer.zero_grad()
                    loss_evaled, clique_loss_val, ae_val, kl_val, weight_val, encoder_out = self._compute_losses(batch_tensor)
                    loss_evaled.backward()
                    self.optimizer.step()
                    bt_loss += float(loss_evaled.item())
        self.optimized = True
