import json

# File path for the JSON data
file_path = "emotion_recognition_results.json"


# Initialize a counter for correct predictions
total_predictions = 0
correct_count = 0

# Load JSON data from file
with open(file_path, 'r') as file:
    data_dict = json.load(file)

# Iterate over each entry in the JSON
for actor, files in data_dict.items():
    for filename, info in files.items():
        # Get the predicted emotion
        predicted_emotion = info["most_likely_emotion"]
        total_predictions += 1
        # Extract the emotion from the filename
        filename_emotion = filename.split('_')[-1].split('.')[0]
        
        # Determine if predicted emotion matches the filename emotion
        if predicted_emotion.lower() == filename_emotion.lower():
            correct_count += 1

# Output the number of correct predictions
print(f"Number of correct predictions: {correct_count}")
print(correct_count / total_predictions)
