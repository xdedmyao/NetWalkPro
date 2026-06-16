import tensorflow as tf

class DenoisingAE(tf.keras.Model):
    def __init__(self, dimensions, activation=tf.nn.sigmoid, lamb=0.01):
        """
        dimensions: list, e.g. [input_dim, hidden_dim]
        """
        super().__init__()
        self.dimensions = dimensions
        self.activation = activation
        self.lamb = lamb

        self.encoder_weights = []
        self.encoder_biases = []
        self.decoder_weights = []
        self.decoder_biases = []

        # 初始化编码器
        for i in range(len(dimensions) - 1):
            W = tf.Variable(tf.random.normal([dimensions[i], dimensions[i+1]]) * 0.01, trainable=True)
            b = tf.Variable(tf.zeros([dimensions[i+1]]), trainable=True)
            self.encoder_weights.append(W)
            self.encoder_biases.append(b)

        # 初始化解码器
        for i in range(len(dimensions) - 1, 0, -1):
            W = tf.Variable(tf.random.normal([dimensions[i], dimensions[i-1]]) * 0.01, trainable=True)
            b = tf.Variable(tf.zeros([dimensions[i-1]]), trainable=True)
            self.decoder_weights.append(W)
            self.decoder_biases.append(b)

    def call(self, x, corrupt_prob=0.0, beta=1.0, rho=0.5, gamma=0.01):
        # 添加噪声
        current = x * (1 - corrupt_prob) + tf.random.uniform(tf.shape(x)) * corrupt_prob

        weight_decay_J = 0.0

        # 编码器前向
        for W, b in zip(self.encoder_weights, self.encoder_biases):
            current = self.activation(tf.matmul(current, W) + b)
            weight_decay_J += (self.lamb / 2.0) * tf.reduce_mean(tf.square(W))

        encoder_out = current

        # 解码器前向
        for W, b in zip(self.decoder_weights, self.decoder_biases):
            current = self.activation(tf.matmul(current, W) + b)
            weight_decay_J += (self.lamb / 2.0) * tf.reduce_mean(tf.square(W))

        reconstruction = current

        # KL sparsity
        rhohats = tf.reduce_mean(encoder_out, axis=0)
        epsilon = 1e-8
        rhohats_safe = tf.clip_by_value(rhohats, epsilon, 1 - epsilon)
        kl = tf.reduce_mean(rho * tf.math.log(rho / rhohats_safe) + (1 - rho) * tf.math.log((1 - rho) / (1 - rhohats_safe)))

        ae_loss = (gamma / 2.0) * tf.reduce_mean(tf.square(reconstruction - x))
        kl_loss = beta * kl
        cost = ae_loss + kl_loss + weight_decay_J

        return {
            "encoder_out": encoder_out,
            "reconstruction": reconstruction,
            "cost": cost,
            "ae_loss": ae_loss,
            "kl_loss": kl_loss,
            "weight_decay_J": weight_decay_J,
            "W_list": self.encoder_weights,
            "b_list": self.encoder_biases
        }
