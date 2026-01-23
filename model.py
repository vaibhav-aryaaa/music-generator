import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from music21 import instrument, note, stream, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization

def load_notes():
  with open("data/notes","rb") as f:
    return pickle.load(f)

def build_model(input_shape, vocab_size):
  model=Sequential([
      LSTM(512, return_sequences=True, input_shape=input_shape),
      Dropout(0.3),
      LSTM(512, return_sequences=True),
      Dropout(0.3),
      LSTM(512),
      BatchNormalization(),
      Dense(256, activation='relu'),
      Dropout(0.3),
      Dense(vocab_size, activation='softmax')
  ])
  model.compile(loss="categorical_crossentropy", optimizer="adam")
  return model

def prepare_input(notes, seq_len=100):
    pitch_names = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(pitch_names)}

    sequences = []

    for i in range(len(notes) - seq_len):
        seq = notes[i:i + seq_len]
        sequences.append([note_to_int[n] for n in seq])

    network_input = np.reshape(
        sequences,
        (len(sequences), seq_len, 1)
    )

    network_input = network_input / float(len(pitch_names))

    return network_input, pitch_names

def sample(preds,temperature=0.9):
  preds=np.log(preds+ 1e-8) / temperature
  preds=np.exp(preds) / np.sum(np.exp(preds))
  return np.random.choice(len(preds), p=preds)

def generate_music(model, net_input, pitch_names, temperature=0.9):
  int_to_note={i: n for i, n in enumerate(pitch_names)}

  start=np.random.randint(0, len(net_input)-1)
  pattern=net_input[start].reshape(-1).tolist()

  output=[]

  for _ in range(300):
    inp=np.reshape(pattern,(1,len(pattern),1))
    pred=model.predict(inp,verbose=0)[0]
    index=sample(pred,temperature)
    output.append(int_to_note[index])

    pattern.append(index)
    pattern=pattern[1:]

  return output

def create_midi(notes, filename="generated.mid"):
    offset = 0
    output_notes = []

    for n in notes:
        if "." in n:
            chord_notes = []
            for x in n.split("."):
                new_note = note.Note(int(x))
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)

            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        elif n.isdigit():
            new_note = note.Note(int(n))
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        else:
            new_note = note.Note(n)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    stream.Stream(output_notes).write("midi", fp=filename)
