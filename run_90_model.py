import tensorflow as tf
from tensorflow import keras
import numpy as np
import librosa
import os
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

def preprocess_audio(file_path, sr=16000, duration=1.094):
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    target_length = 17496
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    audio = audio.reshape(1, 17496, 1, 1)
    return audio

def predict(model, audio_file):
    processed_data = preprocess_audio(audio_file)
    predictions = model.predict(processed_data, verbose=0)
    return predictions

# Define all possible emotion labels
all_emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise', 'Calm']
training_data_path = '/home/anrric/Desktop/emotion-recognition-using-speech/data/training'

# Get all actor folders
actor_folders = [f for f in os.listdir(training_data_path) if os.path.isdir(os.path.join(training_data_path, f))]

all_predictions = []
all_true_labels = []

for test_actor in tqdm(actor_folders, desc="Processing actors"):
    # Load the model for each iteration
    model = keras.models.load_model('cnn_model_2.1_.h5')
    
    test_predictions = []
    test_true_labels = []
    
    # Test on the left-out actor
    test_actor_path = os.path.join(training_data_path, test_actor)
    for audio_file in os.listdir(test_actor_path):
        if audio_file.endswith('.wav'):
            file_path = os.path.join(test_actor_path, audio_file)
            try:
                prediction = predict(model, file_path)[0]
                predicted_emotion = all_emotion_labels[np.argmax(prediction)]
                true_emotion = next((label for label in all_emotion_labels if label.lower() in audio_file.lower()), None)
                
                if true_emotion:
                    test_predictions.append(predicted_emotion)
                    test_true_labels.append(true_emotion)
                else:
                    print(f"Warning: Could not determine true emotion for file {audio_file}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    all_predictions.extend(test_predictions)
    all_true_labels.extend(test_true_labels)

    # Get the unique labels present in this actor's data
    unique_labels = sorted(set(test_true_labels + test_predictions))

    # Print results for this actor
    print(f"\nResults for {test_actor}:")
    try:
        print(classification_report(test_true_labels, test_predictions, target_names=unique_labels, zero_division=0))
    except Exception as e:
        print(f"Error generating classification report: {str(e)}")
        print("True labels:", set(test_true_labels))
        print("Predicted labels:", set(test_predictions))

# Get the unique labels present in all the data
unique_labels = sorted(set(all_true_labels + all_predictions))

# Print overall results
print("\nOverall Results:")
try:
    print(classification_report(all_true_labels, all_predictions, target_names=unique_labels, zero_division=0))
except Exception as e:
    print(f"Error generating overall classification report: {str(e)}")
    print("True labels:", set(all_true_labels))
    print("Predicted labels:", set(all_predictions))

# Print confusion matrix
try:
    cm = confusion_matrix(all_true_labels, all_predictions, labels=unique_labels)
    print("\nConfusion Matrix:")
    print(cm)

    # Calculate overall accuracy
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"\nOverall Accuracy: {accuracy:.2%}")
except Exception as e:
    print(f"Error generating confusion matrix: {str(e)}")
