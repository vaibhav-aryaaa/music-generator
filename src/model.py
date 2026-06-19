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
