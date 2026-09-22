import unittest
import numpy as np
import tensorflow as tf
from src.generate import top_p_sampling

class TestSampling(unittest.TestCase):
    def test_top_p_sampling_returns_int(self):
        """Test that top_p_sampling returns a standard integer."""
        logits = tf.constant([1.0, 2.0, 3.0, 4.0, 0.5])
        next_token = top_p_sampling(logits, p=0.9, temperature=1.0)
        self.assertIsInstance(next_token, int)
        self.assertTrue(0 <= next_token < len(logits))

    def test_sampling_low_temperature(self):
        """Test that with extremely low temperature, it picks the highest logit value (deterministic argmax)."""
        logits = tf.constant([0.1, 10.0, 0.2, 0.3, 0.1])
        # Run multiple times to verify determinism
        selections = [top_p_sampling(logits, p=1.0, temperature=0.001) for _ in range(10)]
        # Index 1 has the largest logit (10.0), it should be selected every time
        for selection in selections:
            self.assertEqual(selection, 1)

    def test_sampling_cutoff_with_p(self):
        """Test that top-p cuts off low probability tokens."""
        # Logits where one is huge, one is medium, and others are tiny
        logits = tf.constant([10.0, 5.0, -100.0, -100.0, -100.0])
        # With p=0.999 and temperature=1.0, only the first two indices should ever be selected
        selections = [top_p_sampling(logits, p=0.99, temperature=1.0) for _ in range(50)]
        for selection in selections:
            self.assertIn(selection, [0, 1])

if __name__ == "__main__":
    unittest.main()
