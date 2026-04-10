# Emotion-recognition
 Speech Emotion Recognition system. It converts audio files into numbers and teaches a neural network to recognize feelings in the voice.

## 1. Feature Extraction (extract_features)
Computers can’t "hear" audio, so librosa converts waves into mathematical patterns:

MFCCs: Captures the "texture" of the speech (the shape of the vocal tract).

Chroma: Represents the pitch and harmonic content.

Mel-spectrogram: Processes frequencies the way human ears do.

## 2. Data Mapping (load_data)
The RAVDESS dataset uses a naming convention (e.g., 03-01-04...). This function:

Decodes: Pulls the emotion code (the 3rd number) to label the data (e.g., 03 = happy).

Splits: Sets aside 20% of the data (test_size=0.2) to test the model on audio it has never "heard" before.

## 3. The Brain (MLPClassifier)
This is a Multi-Layer Perceptron, a type of Neural Network.

Training: It looks at the features and tries to guess the emotion, correcting itself 500 times (max_iter=500).

Saving: joblib.dump saves the trained "brain" as a file. This is the file you upload to Hugging Face Spaces so your app can make predictions instantly.
