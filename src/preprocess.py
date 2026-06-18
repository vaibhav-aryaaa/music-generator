import os
import pickle
import symusic
from tqdm import tqdm
from tokenizer import get_tokenizer

# Paths
WORKSPACE = "/Users/vaibhavarya/Developer/music-generator"
DATA_DIR = os.path.join(WORKSPACE, "data", "midi_dataset")
SAVE_PATH = os.path.join(WORKSPACE, "data", "tokenized_sequences.pkl")

# Style prefix mapping to tokenizer vocab strings
STYLE_PREFIXES = {
    "jigs": "[JIG]_None",
    "reels": "[REEL]_None",
    "waltzes": "[WALTZ]_None",
    "hornpipes": "[HORNPIPE]_None"
}

def main():
    # Load REMI tokenizer
    print("Loading REMI tokenizer...")
    tokenizer = get_tokenizer()
    vocab = tokenizer.vocab
    
    # Check that all style prefixes are in the vocab
    for category, token_str in STYLE_PREFIXES.items():
        if token_str not in vocab:
            raise ValueError(f"Style token '{token_str}' not found in tokenizer vocabulary!")
    
    sequences = []
    skipped_count = 0
    total_processed = 0

    print("Preprocessing MIDI files by category...")
    
    for category, prefix_token in STYLE_PREFIXES.items():
        category_dir = os.path.join(DATA_DIR, category)
        if not os.path.exists(category_dir):
            print(f"Warning: Category directory {category_dir} does not exist. Skipping.")
            continue
            
        prefix_id = vocab[prefix_token]
        files = [f for f in os.listdir(category_dir) if f.endswith(".mid") or f.endswith(".midi")]
        print(f"Processing category '{category}' ({len(files)} files)...")
        
        for filename in tqdm(files):
            file_path = os.path.join(category_dir, filename)
            try:
                # Load MIDI file
                score = symusic.Score(file_path)
                
                # If there are no tracks, skip
                if len(score.tracks) == 0:
                    skipped_count += 1
                    continue
                
                # Merge all tracks into track 0
                track0 = score.tracks[0]
                for track in score.tracks[1:]:
                    for note in track.notes:
                        track0.notes.append(note)
                
                # Sort notes chronologically
                track0.notes.sort(key=lambda x: x.time)
                score.tracks = [track0]
                
                # Tokenize the score
                tokens_list = tokenizer(score)
                if len(tokens_list) == 0:
                    skipped_count += 1
                    continue
                    
                # Extract token IDs
                ids = tokens_list[0].ids
                
                # Prepend the style prefix token ID
                ids_with_prefix = [prefix_id] + ids
                
                sequences.append({
                    "ids": ids_with_prefix,
                    "category": category,
                    "filename": filename
                })
                
                total_processed += 1
                
            except Exception as e:
                # Skip corrupted files silently or with a debug print
                skipped_count += 1
                
    # Print summary
    print(f"\nPreprocessing complete!")
    print(f"Successfully processed {total_processed} files.")
    print(f"Skipped {skipped_count} invalid or empty files.")
    print(f"Total sequences collected: {len(sequences)}")
    
    # Save the processed data
    print(f"Saving tokenized sequences to {SAVE_PATH}...")
    save_data = {
        "sequences": sequences,
        "vocab_size": len(vocab),
        "style_prefixes": {cat: vocab[tok] for cat, tok in STYLE_PREFIXES.items()}
    }
    
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(save_data, f)
    print("Dataset saved successfully!")

if __name__ == "__main__":
    main()
