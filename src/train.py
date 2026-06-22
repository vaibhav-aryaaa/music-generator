import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.model import build_model

# Configuration
WORKSPACE = "/Users/vaibhavarya/Developer/music-generator"
DATA_PATH = os.path.join(WORKSPACE, "data", "tokenized_sequences.pkl")
MODEL_SAVE_DIR = os.path.join(WORKSPACE, "models")
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.keras")

# Hyperparameters
SEQ_LEN = 256        # Context window length for the transformer
EMBED_DIM = 256      # Embedding dimension size
NUM_HEADS = 8        # Number of self-attention heads
FF_DIM = 512         # Feed-forward network hidden size
NUM_LAYERS = 4       # Number of stacked transformer blocks
BATCH_SIZE = 64      # Batch size
EPOCHS = 20          # Maximum training epochs
VAL_SPLIT = 0.1      # Validation split ratio

def prepare_dataset(data_path, seq_len=SEQ_LEN):
    """
    Loads tokenized sequences and slices them into fixed-length chunks.
    For each chunk, the style conditioning token is prepended to maintain
    autoregressive style conditioning.
    """
    print(f"Loading tokenized dataset from {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
        
    raw_sequences = data["sequences"]
    vocab_size = data["vocab_size"]
    
    inputs = []
    targets = []
    
    # We want chunks of size seq_len. 
    # Since input is X (length seq_len - 1) and target is Y (length seq_len - 1),
    # we need slice of size seq_len - 1.
    slice_len = seq_len - 1
    
    print("Slicing sequences into chunks...")
    for item in raw_sequences:
        ids = item["ids"]
        style_token_id = ids[0]  # The first token is always the style token (e.g. [JIG])
        content_ids = ids[1:]    # The rest of the song
        
        # Slice the content into chunks
        for i in range(0, len(content_ids) - slice_len, slice_len // 2):  # 50% overlap for data augmentation
            chunk = content_ids[i:i + slice_len]
            if len(chunk) < slice_len:
                continue
                
            # Prepend style token to input X
            # Input X is: [STYLE] + chunk[:-1]
            x = [style_token_id] + chunk[:-1]
            
            # Target Y is: chunk
            # Shifted by 1 (predicting chunk[t] given style + chunk[:t])
            y = chunk
            
            inputs.append(x)
            targets.append(y)
            
    inputs = np.array(inputs, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)
    
    print(f"Created {len(inputs)} training slices of length {seq_len}.")
    return inputs, targets, vocab_size

def main():
    # Verify GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow is using GPU acceleration: {gpus}")
    else:
        print("TensorFlow is running on CPU.")

    # Prepare inputs and targets
    X, Y, vocab_size = prepare_dataset(DATA_PATH, SEQ_LEN)
    
    # Shuffle and split into train/val
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]
    
    split_idx = int(num_samples * (1 - VAL_SPLIT))
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_val, Y_val = X[split_idx:], Y[split_idx:]
    
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Build model
    print("Assembling Music Transformer Decoder...")
    model = build_model(
        vocab_size=vocab_size,
        maxlen=SEQ_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        dropout_rate=0.1
    )
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        metrics=["accuracy"]
    )
    
    # Ensure save directory exists
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    # Start training
    print(f"Starting model training for {EPOCHS} epochs...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    print("Training finished!")
    print(f"Best model weights saved to: {BEST_MODEL_PATH}")

if __name__ == "__main__":
    main()
