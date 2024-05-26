**Dataset can be accessed here: "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"**

# Emotion Recognition from Speech Using Deep Learning

# 1. Overview:
This project aims to build a deep learning model that can recognize emotions from speech using the RAVDESS dataset. The RAVDESS dataset consists of short clips of actors speaking different emotions. Emotions are categorized into 8 classes: neutral, calm, happy, sad, angry, fearful, disgust, and surprised. The model leverages Mel Spectrogram features extracted from audio files to classify the emotions.

# 2. Features:
* Dataset Handling: Uses the RAVDESS dataset, which provides actors speaking scripted emotions in English.
* Preprocessing: Extracts Mel Spectrogram features from audio files using Librosa.
* Model Building: Constructs a Convolutional Neural Network (CNN) using Keras to classify emotions.
* Evaluation: Evaluates the model's performance using metrics such as accuracy and confusion matrix.
* Visualization: Displays waveplots and Mel Spectrograms for different emotions.

# 3. Libraries Used:
* Librosa for audio feature extraction.
* Matplotlib for plotting visualizations.
* TensorFlow and Keras for building and training the deep learning model.
* Scikit-learn for model evaluation and metrics.

# 4. Project Structure:
* Data Preparation: Extracts emotion labels and processes audio files.
* Feature Extraction: Computes Mel Spectrogram features and preprocesses them for training.
* Model Development: Constructs a CNN model architecture for emotion classification.
* Training and Evaluation: Trains the model, evaluates its performance, and visualizes results.

# 5. Results:
Achieves an accuracy of [accuracy] on the test set.

Displays a confusion matrix and classification report for model evaluation.

Provides predictions and actual labels for test set samples.
