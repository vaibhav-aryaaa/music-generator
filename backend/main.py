import os
import pickle
import tempfile
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from miditok import REMI

from src.model import TokenAndPositionEmbedding, TransformerBlock, CausalSelfAttention
from src.generate import top_p_sampling

model = None
tokenizer = None
style_prefixes = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle startup loading of model and tokenizer
    and cleanup on shutdown.
    """
    global model, tokenizer, style_prefixes
    
    # Paths
    workspace = "/Users/vaibhavarya/Developer/music-generator"
    model_path = os.path.join(workspace, "models", "best_model.keras")
    tokenizer_path = os.path.join(workspace, "models", "tokenizer.json")
    dataset_metadata_path = os.path.join(workspace, "data", "tokenized_sequences.pkl")
    
    # Load Tokenizer
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run tokenizer setup first.")
    tokenizer = REMI(params=tokenizer_path)
    
    # Load Style Prefixes mapping
    if not os.path.exists(dataset_metadata_path):
        raise FileNotFoundError(f"Dataset metadata not found at {dataset_metadata_path}. Run preprocessing first.")
    with open(dataset_metadata_path, "rb") as f:
        data = pickle.load(f)
    style_prefixes = data["style_prefixes"]
    
    # Load Model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}. Run training first.")
        
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
            "TransformerBlock": TransformerBlock,
            "CausalSelfAttention": CausalSelfAttention
        }
    )
    print("FastAPI Backend: Model, tokenizer, and style maps loaded successfully!")
    yield
    print("FastAPI Backend: Shutting down...")

# Initialize FastAPI
app = FastAPI(
    title="🎵 AI Music Generator API",
    description="FastAPI Backend for generating style-conditioned music using a Keras Transformer.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    mood: str          # jigs, reels, waltzes, hornpipes
    length: int = 300  # Number of tokens to generate
    temp: float = 1.0  # Creativity temperature
    top_p: float = 0.9 # Nucleus sampling threshold

@app.get("/")
def read_root():
    return {"message": "AI Music Generator API is running!", "available_moods": list(style_prefixes.keys())}

@app.post("/generate")
def generate_midi(req: GenerateRequest):
    """
    Generates a MIDI file based on the requested style and parameters,
    and returns the file directly as a download response.
    """
    global model, tokenizer, style_prefixes
    
    if req.mood not in style_prefixes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style/mood '{req.mood}'. Available: {list(style_prefixes.keys())}"
        )
        
    style_id = style_prefixes[req.mood]
    
    try:
        # Generate token sequence autoregressively
        sequence = [style_id]
        context_len = 255  # SEQ_LEN - 1 (Window size 256)
        
        for _ in range(req.length):
            # Form input window prepended with the style token
            if len(sequence) > context_len:
                inputs = [style_id] + sequence[-context_len:]
            else:
                inputs = sequence
                
            inputs_tensor = tf.expand_dims(inputs, 0)
            predictions = model(inputs_tensor, training=False)
            logits = predictions[0, -1, :]
            
            next_token = top_p_sampling(logits, p=req.top_p, temperature=req.temp)
            sequence.append(next_token)
            
        # Extract token IDs without the style prefix
        midi_ids = sequence[1:]
        
        # Decode back to a symusic Score object
        score = tokenizer.decode([midi_ids])
        
        # Create a temporary file to save the MIDI data
        temp_midi = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
        score.dump_midi(temp_midi.name)
        temp_midi.close()
        
        return FileResponse(
            temp_midi.name,
            media_type="audio/midi",
            filename=f"generated_{req.mood}.mid"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/synthesize")
def synthesize_audio(req: GenerateRequest):
    """
    Generates MIDI, then synthesizes it to a WAV audio file using midi2audio/fluidsynth.
    Requires fluidsynth to be installed on the host system.
    """
    # Import midi2audio inside the endpoint so it is only called if requested
    try:
        from midi2audio import FluidSynth
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Package 'midi2audio' not installed. Run './venv/bin/pip install midi2audio'."
        )
        
    # Check if fluidsynth binary exists on system
    import shutil
    if not shutil.which("fluidsynth"):
        raise HTTPException(
            status_code=503,
            detail="Fluidsynth system binary not found. Please install it by running 'brew install fluidsynth' in your terminal."
        )
        
    # Generate MIDI file first
    midi_response = generate_midi(req)
    temp_midi_path = midi_response.path
    
    try:
        # Create a temporary file to save the synthesized WAV
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav.close()
        
        # Synthesize using midi2audio FluidSynth
        sf2_path = "/opt/homebrew/share/soundfonts/FluidR3_GM.sf2"
        if not os.path.exists(sf2_path):
            # Check other common paths
            fallback_paths = [
                "/usr/share/sounds/sf2/FluidR3_GM.sf2",
                "/usr/local/share/soundfonts/FluidR3_GM.sf2"
            ]
            for path in fallback_paths:
                if os.path.exists(path):
                    sf2_path = path
                    break
            else:
                # If no soundfont is found, raise error
                raise HTTPException(
                    status_code=404,
                    detail="No soundfont (.sf2) file found. Please install a soundfont, e.g., 'brew install fluid-soundfont'."
                )
                
        fs = FluidSynth(sound_font=sf2_path)
        fs.midi_to_audio(temp_midi_path, temp_wav.name)
        
        if os.path.exists(temp_midi_path):
            os.remove(temp_midi_path)
            
        return FileResponse(
            temp_wav.name,
            media_type="audio/wav",
            filename=f"generated_{req.mood}.wav"
        )
        
    except Exception as e:
        if os.path.exists(temp_midi_path):
            os.remove(temp_midi_path)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")
