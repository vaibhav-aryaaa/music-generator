import unittest
import tensorflow as tf
from src.model import (
    CausalSelfAttention,
    TokenAndPositionEmbedding,
    TransformerBlock,
    build_model
)

class TestTransformerModel(unittest.TestCase):
    def test_causal_self_attention_shape(self):
        """Test that the CausalSelfAttention layer preserves input shapes."""
        embed_dim = 64
        num_heads = 2
        batch_size = 2
        seq_len = 16
        
        layer = CausalSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        dummy_input = tf.random.uniform((batch_size, seq_len, embed_dim))
        output = layer(dummy_input)
        
        self.assertEqual(output.shape, (batch_size, seq_len, embed_dim))

    def test_token_and_position_embedding_shape(self):
        """Test that the TokenAndPositionEmbedding layer correctly embeds token indices into vectors."""
        maxlen = 128
        vocab_size = 318
        embed_dim = 64
        batch_size = 2
        seq_len = 16
        
        layer = TokenAndPositionEmbedding(maxlen=maxlen, vocab_size=vocab_size, embed_dim=embed_dim)
        dummy_input = tf.random.uniform((batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32)
        output = layer(dummy_input)
        
        self.assertEqual(output.shape, (batch_size, seq_len, embed_dim))

    def test_transformer_block_shape(self):
        """Test that the TransformerBlock layer correctly preserves shapes."""
        embed_dim = 64
        num_heads = 2
        ff_dim = 128
        batch_size = 2
        seq_len = 16
        
        layer = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
        dummy_input = tf.random.uniform((batch_size, seq_len, embed_dim))
        output = layer(dummy_input)
        
        self.assertEqual(output.shape, (batch_size, seq_len, embed_dim))

    def test_full_model_inference_shape(self):
        """Test that the compiled GPT-style model outputs probabilities over the vocab."""
        vocab_size = 318
        maxlen = 256
        embed_dim = 64
        num_heads = 2
        ff_dim = 128
        num_layers = 2
        
        model = build_model(
            vocab_size=vocab_size,
            maxlen=maxlen,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers
        )
        
        batch_size = 2
        seq_len = 32
        dummy_input = tf.random.uniform((batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32)
        output = model(dummy_input)
        
        # Output shape should be (batch_size, seq_len, vocab_size)
        self.assertEqual(output.shape, (batch_size, seq_len, vocab_size))

if __name__ == "__main__":
    unittest.main()
