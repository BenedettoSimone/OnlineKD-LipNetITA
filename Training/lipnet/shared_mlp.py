import tensorflow as tf


class SharedMLP(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),  # Input layer
            tf.keras.layers.Dense(hidden_dim, activation='relu'),  # Hidden layer
            tf.keras.layers.Dense(output_dim)  # Output layer
        ])

    def call(self, x, **kwargs):
        # Forward pass
        return self.mlp(x)
