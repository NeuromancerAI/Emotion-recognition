import os
import soundfile
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def extract_features(file_path, mfcc, chroma, mel):
    """Extracts features from an audio file."""
    with soundfile.SoundFile(file_path) as audio:
        X = audio.read(dtype="float32")
        sample_rate = audio.samplerate
        features = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            features = np.hstack((features, mfccs))
        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma_vals = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            features = np.hstack((features, chroma_vals))
        if mel:
            mel_vals = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            features = np.hstack((features, mel_vals))
    return features

def load_data(data_path, test_size=0.2):
    """Loads the dataset and splits it into training and testing sets."""
    X, y = [], []
    # Emotions from the RAVDESS dataset filenames
    emotions = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }

    for file in os.listdir(data_path):
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            if emotion_code in emotions:
                emotion = emotions[emotion_code]
                feature = extract_features(os.path.join(data_path, file), mfcc=True, chroma=True, mel=True)
                X.append(feature)
                y.append(emotion)

    return train_test_split(np.array(X), y, test_size=test_size, random_state=9)

if __name__ == "__main__":
    # Define the path to your dataset
    # You must download the RAVDESS dataset and place the audio files in this directory
    DATA_PATH = "ravdess_data/"
    os.makedirs(DATA_PATH, exist_ok=True)
    
    print("Loading data... Make sure you've downloaded the RAVDESS dataset into the 'ravdess_data' folder.")
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    if not X_train.any():
        print("Dataset is empty! Please check the DATA_PATH and ensure audio files are present.")
    else:
        print(f"Training model on {len(X_train)} samples...")
        model = MLPClassifier(alpha=0.01, batch_size=256, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        print(f"Model Accuracy: {accuracy*100:.2f}% 🚀")

        # Save the model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/emotion_classifier.joblib")
        print("Trained model saved to 'models/emotion_classifier.joblib'")
