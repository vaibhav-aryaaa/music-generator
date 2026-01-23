# 🎵 AI Music Generator (LSTM + Streamlit)

An AI-powered music generation project that composes piano melodies using a deep learning LSTM model.  
The project includes a Streamlit web interface to generate and download MIDI files interactively.

---

## Features

- 🎹 Piano music generation using LSTM neural networks  
- 🎚️ Temperature control for creativity  
- 🎼 Generates MIDI files  
- ⬇️ Download generated music  
- 🌐 Streamlit-based web UI  
- ☁️ Google Colab + Google Drive compatible  

---

## Model Overview

- Architecture: **Stacked LSTM**
- Framework: **TensorFlow / Keras**
- Input: Sequences of musical notes
- Output: Probability distribution over next notes
- Loss: Categorical Crossentropy
- Optimizer: Adam

The model is trained on MIDI data converted into note/chord sequences using `music21`.

---

## Tech Stack

* **Python**
* **TensorFlow / Keras**
* **Streamlit**
* **music21**
* **NumPy, Pandas**
* **Google Colab**

## Project Structure
```text
music_generator/
│
├── model.py # Model architecture & helper functions
├── app.py # Streamlit web app
├── weights.hdf5 # Trained model weights
├── data/
│ └── notes
├── generate.ipynb # Local generation testing
└── README.md

---

#### Install dependencies
    ``` bash
    pip install streamlit tensorflow music21 numpy pandas pyngrok
    ```
#### Run Streamlit App
    ```bash
    streamlit run app.py
    ```
    To expose the app publicly in Colab, use ngrok or localtunnel.

---

## 🎚️ Temperature Control

The temperature slider controls creativity:

| Temperature      | Behavior                  |
| ---------------- | ------------------------- |
| Low (0.3–0.6)    | Safe, repetitive melodies |
| Medium (0.8–1.0) | Balanced & musical        |
| High (1.1–1.5)   | Creative, unpredictable   |

---

## 🎶 Output

* **Format**: MIDI (.mid)

* **Instrument**: Piano

* **Length**: Configurable (default ~300 notes)

The MIDI file can be played in any DAW or media player supporting MIDI.

    
