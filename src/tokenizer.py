import os
from miditok import REMI, TokenizerConfig

# Special style prefix tokens for conditioning
SPECIAL_TOKENS = ["PAD", "BOS", "EOS", "MASK", "[JIG]", "[REEL]", "[WALTZ]", "[HORNPIPE]"]

def get_tokenizer():
    """
    Creates and configures a REMI tokenizer for music generation.
    REMI (REpresentation for Music Inference) is an event-based representation
    well-suited for Transformers.
    """
    config = TokenizerConfig(
        num_velocities=16,
        use_chords=True,
        use_tempos=True,
        num_tempos=32,
        use_rests=True,
        use_time_signatures=False,
        # Beat resolution: 8 subdivisions per beat (32nd notes resolution)
        beat_res={(0, 4): 8},
        special_tokens=SPECIAL_TOKENS
    )
    
    tokenizer = REMI(config)
    return tokenizer

def save_tokenizer(tokenizer, save_path="models/tokenizer.json"):
    """
    Saves the tokenizer parameters to a JSON file for later reuse during inference.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer parameters successfully saved to {save_path}")

if __name__ == "__main__":
    # Test initialization
    tok = get_tokenizer()
    print("REMI Tokenizer initialized successfully!")
    print("Vocabulary size:", len(tok.vocab))
    print("Special tokens:", tok.special_tokens)
    save_tokenizer(tok)
