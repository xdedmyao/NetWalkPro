import tensorflow as tf
import numpy as np
from scipy.sparse import csgraph
from anomaly import DenoisingAE


class Model:
    def __init__(self, activation, dimension, walk_len, nodeNum,
                 gama, lamb, beta, rho, epoch, batch_size,
                 learning_rate, optimizer_type, corrupt_prob):

        self.activation = activation
        self.dimension = dimension
        self.walk_len = walk_len
        self.nodeNum = nodeNum
        self.gama = gama
        self.lamb = lamb
        self.beta = beta
        self.rho = rho
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.corrupt_prob = corrupt_prob

        self.ae_model = DenoisingAE(dimension, activation=activation, lamb=lamb)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)

    def forward_ae(self, x):
        return self.ae_model(x, corrupt_prob=self.corrupt_prob,
                             beta=self.beta, rho=self.rho, gamma=self.gama)

    def clique_embedding_loss(self, batch):
        result = self.forward_ae(batch)
        encoder_out = result["encoder_out"]
        ae_cost = result["cost"]

        phi = np.ones((self.walk_len, self.walk_len)) - np.eye(self.walk_len)
        L = tf.constant(csgraph.laplacian(phi, normed=False), dtype=tf.float32)

        code = tf.reshape(encoder_out, [-1, self.walk_len, self.dimension[-1]])
        code_t = tf.transpose(code, perm=[0, 2, 1])

        left = tf.einsum("aij,jk->aik", code_t, L)
        mul = tf.einsum("aij,ajk->aik", left, code)
        trace_vals = tf.stack([tf.linalg.trace(mul[i]) for i in range(mul.shape[0])])
        clique_loss = tf.reduce_mean(trace_vals)

        total_loss = ae_cost + clique_loss

        return total_loss, clique_loss, result["ae_loss"], result["kl_loss"], result["weight_decay_J"]

    def fit(self, data_train):
        # 确保 shape: [batch_size, input_dim]
        data_train = data_train.T.astype(np.float32)

        for epoch in range(self.epoch):
            nbatch = data_train.shape[0] // self.batch_size
            for i in range(nbatch):
                batch = data_train[i * self.batch_size:(i + 1) * self.batch_size]

                with tf.GradientTape() as tape:
                    loss, clique_loss, ae_loss, kl_loss, w_loss = self.clique_embedding_loss(batch)
                grads = tape.gradient(loss, self.ae_model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.ae_model.trainable_variables))

                print(f"Epoch {epoch} batch {i} | Loss: {loss.numpy():.4f} "
                      f"| Clique: {clique_loss.numpy():.4f} | AE: {ae_loss.numpy():.4f} | KL: {kl_loss.numpy():.4f}")

    def feedforward_autoencoder(self, data):
        data = data.astype(np.float32)
        result = self.forward_ae(data)
        return result["encoder_out"].numpy()
