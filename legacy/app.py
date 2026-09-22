import streamlit as st
import numpy as np
import tensorflow as tf
import os

from model import (
    load_notes,
    prepare_input,
    build_model,
    generate_music,
    create_midi
)

st.set_page_config(
    page_title="🎵 AI Music Generator",
    page_icon="🎹",
    layout="centered"
)

st.title("🎵 AI Music Generator")
st.write("Generate piano music using a deep learning LSTM model")

@st.cache_resource
def load_everything():
    notes = load_notes()
    net_input, pitch_names = prepare_input(notes)

    model = build_model(
        input_shape=(net_input.shape[1], net_input.shape[2]),
        vocab_size=len(pitch_names)
    )

    model.load_weights("weights.hdf5", by_name=True, skip_mismatch=True)
    return model, net_input, pitch_names


model, net_input, pitch_names = load_everything()

temperature = st.slider(
    "🎚️ Temperature (Creativity)",
    0.2, 1.5, 0.9, 0.1
)

length = st.slider("🎶 Music Length", 100, 600, 300, 50)

if st.button("🎼 Generate Music"):
    import time
    np.random.seed(int(time.time()))

    with st.spinner("Composing music... 🎶"):
        generated_notes = generate_music(
            model,
            net_input,
            pitch_names,
            temperature
        )

        output_file = "generated.mid"
        create_midi(generated_notes, output_file)

    st.success("🎉 Music generated!")

    with open(output_file, "rb") as f:
        st.download_button(
            "⬇️ Download MIDI",
            f,
            file_name="ai_music.mid",
            mime="audio/midi"
        )
