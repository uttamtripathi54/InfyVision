# models/transformer_model.py
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from config import TRANSFORMER_DIM, TRANSFORMER_HEADS, TRANSFORMER_LAYERS, DROPOUT_RATE, EPOCHS, BATCH_SIZE
from utils.logger import setup_logger

logger = setup_logger(__name__)

class TransformerBlock(layers.Layer):
    """Single Transformer encoder block."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerModel:
    """Simple Transformer encoder for time series."""
    def __init__(self, input_shape, embed_dim=TRANSFORMER_DIM, num_heads=TRANSFORMER_HEADS,
                 num_layers=TRANSFORMER_LAYERS, rate=DROPOUT_RATE, name='transformer'):
        self.input_shape = input_shape  # (timesteps, features)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.rate = rate
        self.name = name
        self.model = None
        self.history = None

    def build(self, output_steps=1):
        """Build Transformer model.
        
        Args:
            output_steps: number of prediction steps (1 for next day, 5 for next 5 days)
        """
        inputs = layers.Input(shape=self.input_shape)
        # Project to embed_dim
        x = layers.Dense(self.embed_dim)(inputs)
        # Add positional encoding
        pos_encoding = self.add_positional_encoding(x)
        x = layers.Add()([x, pos_encoding])
        
        # Stack transformer blocks
        for _ in range(self.num_layers):
            attn_output = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)(x, x)
            attn_output = layers.Dropout(self.rate)(attn_output)
            x = layers.Add()([x, attn_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            
            ffn = layers.Dense(self.embed_dim*4, activation='relu')(x)
            ffn = layers.Dense(self.embed_dim)(ffn)
            ffn = layers.Dropout(self.rate)(ffn)
            x = layers.Add()([x, ffn])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(self.rate)(x)
        outputs = layers.Dense(output_steps)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        self.output_steps = output_steps
        logger.info(f"Transformer model built with {output_steps} output steps.")
        return model

    def add_positional_encoding(self, x):
        """Sinusoidal positional encoding."""
        import numpy as np
        timesteps = int(x.shape[1])
        pos_encoding = np.zeros((timesteps, self.embed_dim))
        position = np.arange(0, timesteps)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        return tf.constant(pos_encoding[np.newaxis, :, :], dtype=tf.float32)

    def train(self, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
        if self.model is None:
            self.build()
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        logger.info("Transformer training completed.")
        return self.history

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def save(self, path):
        self.model.save(path)
        logger.info(f"Transformer model saved to {path}")

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        logger.info(f"Transformer model loaded from {path}")
        