import os
import argparse
import pickle
import numpy as np
import tensorflow as tf
from miditok import REMI
from model import TokenAndPositionEmbedding, TransformerBlock, CausalSelfAttention

# Paths
WORKSPACE = "/Users/vaibhavarya/Developer/music-generator"
TOKENIZED_DATA_PATH = os.path.join(WORKSPACE, "data", "tokenized_sequences.pkl")
MODEL_PATH = os.path.join(WORKSPACE, "models", "best_model.keras")
TOKENIZER_PATH = os.path.join(WORKSPACE, "models", "tokenizer.json")

def top_p_sampling(logits, p=0.9, temperature=1.0):
    """
    Applies temperature scaling and top-p (nucleus) sampling to logits.
    """
    # Apply temperature scaling
    logits = logits / max(temperature, 1e-8)
    
    # Convert to probabilities
    probs = tf.nn.softmax(logits).numpy()
    
    # Sort probabilities in descending order
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Calculate cumulative probabilities
    cumulative_probs = np.cumsum(sorted_probs)
    
    # Find the cutoff where cumulative probability exceeds p
    # Keep at least the top 1 candidate
    cutoff_idx = np.where(cumulative_probs >= p)[0]
    if len(cutoff_idx) > 0:
        cutoff = cutoff_idx[0] + 1
    else:
        cutoff = len(sorted_probs)
        
    # Truncate
    top_indices = sorted_indices[:cutoff]
    top_probs = sorted_probs[:cutoff]
    
    # Re-normalize probabilities
    top_probs = top_probs / np.sum(top_probs)
    
    # Draw next token from the selected subset
    next_token = np.random.choice(top_indices, p=top_probs)
    return int(next_token)

def generate_music(model, tokenizer, style_id, gen_length=300, seq_len=256, temperature=1.0, top_p=0.9):
    """
    Generates a sequence of token IDs autoregressively.
    """
    print(f"Generating sequence of length {gen_length}...")
    
    # Start sequence with just the style ID
    sequence = [style_id]
    
    # Max context sequence size we can feed the model is seq_len
    # Since we prepend the style token, the history window size is seq_len - 1
    context_len = seq_len - 1
    
    for step in range(gen_length):
        # Prepare input: keep the last context_len tokens and prepend style prefix
        if len(sequence) > context_len:
            # Drop older elements, keep recent context, prepend style ID
            inputs = [style_id] + sequence[-context_len:]
        else:
            inputs = sequence
            
        # Convert to tensor: shape (1, seq_len)
        inputs_tensor = tf.expand_dims(inputs, 0)
        
        # Get model prediction log probabilities (logits)
        # Predictions shape: (1, seq_len, vocab_size)
        predictions = model(inputs_tensor, training=False)
        
        # Extract predictions for the last token in the sequence: shape (vocab_size,)
        logits = predictions[0, -1, :]
        
        # Draw next token via top-p sampling
        next_token = top_p_sampling(logits, p=top_p, temperature=temperature)
        
        # Append next token
        sequence.append(next_token)
        
        if (step + 1) % 50 == 0:
            print(f"  Generated {step + 1}/{gen_length} tokens...")
            
    # Return sequence without the starting style prefix
    return sequence[1:]

def main():
    parser = argparse.ArgumentParser(description="Autoregressive Music Transformer Generator")
    parser.add_argument("--mood", type=str, default="jigs", choices=["jigs", "reels", "waltzes", "hornpipes"],
                        help="Musical style/mood prefix")
    parser.add_argument("--length", type=int, default=300, help="Number of tokens to generate")
    parser.add_argument("--temp", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) threshold")
    parser.add_argument("--output", type=str, default="generated.mid", help="Output MIDI file path")
    args = parser.parse_args()

    # Verify model and tokenizer files exist
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run training first.")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}. Run tokenizer first.")

    # Load style prefixes map from dataset
    print(f"Loading style prefix mapping from dataset...")
    with open(TOKENIZED_DATA_PATH, "rb") as f:
        data = pickle.load(f)
    style_prefixes = data["style_prefixes"]
    
    if args.mood not in style_prefixes:
        raise ValueError(f"Selected style '{args.mood}' not recognized in dataset style prefixes.")
    style_id = style_prefixes[args.mood]
    print(f"Selected style: '{args.mood}' (Token ID: {style_id})")

    # Load tokenizer
    print("Loading REMI tokenizer...")
    tokenizer = REMI(params=TOKENIZER_PATH)

    # Load model with custom layers
    print("Loading trained Keras model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
            "TransformerBlock": TransformerBlock,
            "CausalSelfAttention": CausalSelfAttention
        }
    )

    # Generate token sequence
    generated_ids = generate_music(
        model=model,
        tokenizer=tokenizer,
        style_id=style_id,
        gen_length=args.length,
        seq_len=256,
        temperature=args.temp,
        top_p=args.top_p
    )

    # Decode tokens back into MIDI
    print("Decoding generated tokens to MIDI score...")
    try:
        # Pass a list containing the generated ID list (one stream)
        score = tokenizer.decode([generated_ids])
        
        # Save to file
        score.dump_midi(args.output)
        print(f"🎉 Success! Generated music successfully saved to {args.output}")
    except Exception as e:
        print(f"Error decoding tokens to MIDI: {e}")

if __name__ == "__main__":
    main()
