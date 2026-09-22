import unittest
import os
from miditok import REMI
from src.tokenizer import get_tokenizer, SPECIAL_TOKENS

class TestTokenizer(unittest.TestCase):
    def test_tokenizer_initialization(self):
        """Test that get_tokenizer correctly instantiates a REMI tokenizer with custom parameters."""
        tokenizer = get_tokenizer()
        self.assertIsInstance(tokenizer, REMI)
        
        # Check vocab size (REMI tokenizer vocabulary is typically > 200 tokens)
        vocab_size = len(tokenizer.vocab)
        self.assertGreater(vocab_size, 200)

    def test_special_tokens(self):
        """Test that the custom special tokens are registered in the tokenizer vocab."""
        tokenizer = get_tokenizer()
        
        # Check that style tag tokens exist in vocabulary
        for token in ["[JIG]", "[REEL]", "[WALTZ]", "[HORNPIPE]"]:
            self.assertIn(f"{token}_None", tokenizer.vocab)
            # Ensure they map to valid integers
            token_id = tokenizer[f"{token}_None"]
            self.assertIsInstance(token_id, int)

    def test_save_and_load_tokenizer(self):
        """Test saving and loading the tokenizer parameters JSON file."""
        tokenizer = get_tokenizer()
        temp_path = "models/test_tokenizer.json"
        
        # Save tokenizer
        tokenizer.save(temp_path)
        self.assertTrue(os.path.exists(temp_path))
        
        # Load and verify vocab sizes match
        loaded_tokenizer = REMI(params=temp_path)
        self.assertEqual(len(tokenizer.vocab), len(loaded_tokenizer.vocab))
        
        # Clean up temporary test file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    unittest.main()
