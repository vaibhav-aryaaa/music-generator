import tensorflow as tf
from tensorflow.keras import layers

class CausalSelfAttention(layers.Layer):
    """
    Custom Causal Self-Attention layer for Autoregressive Sequence Generation.
    It wraps Keras's MultiHeadAttention with use_causal_mask=True to ensure
    that tokens can only attend to previous tokens in the sequence (preventing
    information leakage from the future).
    """
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-Head Attention layer. Key dimension is usually embed_dim // num_heads
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads
        )
        
        # Layer Normalization and Dropout
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # Self-attention call with query=key=value=x
        # Setting use_causal_mask=True enables causal masking natively in Keras 3
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True,
            training=training
        )
        attn_output = self.dropout(attn_output, training=training)
        
        # Residual connection and normalization
        return self.layernorm(x + attn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        })
        return config

class TokenAndPositionEmbedding(layers.Layer):
    """
    Combines token embeddings and learned positional embeddings.
    For a sequence of tokens, this layer maps each token ID to an embedding vector
    and adds a corresponding learned position vector.
    """
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Token embedding layer
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        
        # Positional embedding layer
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        # Dynamically get sequence length from input
        seq_len = tf.shape(x)[-1]
        
        # Generate position indices [0, 1, 2, ..., seq_len - 1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)
        
        # Get token embeddings
        x = self.token_emb(x)
        
        # Sum both embeddings
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

class TransformerBlock(layers.Layer):
    """
    Standard Transformer Decoder Block.
    It combines CausalSelfAttention with a Feed-Forward Network (FFN)
    and residual connections.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        # Self-attention block (Day 3 Task 1)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout_rate)
        
        # Feed-forward network (FFN) block
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        
        # Output layer norm and dropout
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # Attention block (already handles layernorm + residual internally)
        attn_out = self.attn(x, training=training)
        
        # FFN block
        ffn_out = self.ffn(attn_out)
        ffn_out = self.dropout(ffn_out, training=training)
        
        # Second residual connection and normalization
        return self.layernorm(attn_out + ffn_out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
        })
        return config

def build_model(vocab_size, maxlen, embed_dim, num_heads, ff_dim, num_layers, dropout_rate=0.1):
    """
    Assembles the complete GPT-style Decoder-Only Transformer.
    """
    inputs = layers.Input(shape=(None,), dtype=tf.int32)
    
    # Token and Position Embeddings (Day 3 Task 2)
    x = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)(inputs)
    
    # Stack of Transformer Decoder Blocks
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)
        
    # Dense Softmax Projection over the vocabulary
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    # Test model building and compilation
    print("Assembling dummy Transformer Decoder...")
    model = build_model(
        vocab_size=318,     # Matching our tokenizer vocabulary
        maxlen=1024,        # Maximum context sequence length
        embed_dim=256,      # Embedding dimension
        num_heads=4,        # Attention heads
        ff_dim=512,         # Feed-forward hidden dimension
        num_layers=4        # Number of stacked decoder blocks
    )
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    model.summary()
    print("\nModel assembled and compiled successfully!")
