import librosa
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Define a function to extract audio features
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    features = np.concatenate([mfccs.mean(axis=1), chroma.mean(axis=1)])
    return features

# Define a function to predict the class of an audio sample
def predict(audio_path):
    features = extract_features(audio_path)
    features = np.expand_dims(features, axis=0)
    pred = model.predict(features)[0]
    if pred < 0.5:
        return 'human'
    else:
        return 'AI'

# Test the function on a sample audio file
audio_path = 'sample.wav'
print(predict(audio_path))
