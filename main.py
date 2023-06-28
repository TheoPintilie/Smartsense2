import numpy as np
import librosa
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("smartsense_model.h5")

# The bird species vector
bird_species = ["ana", "chira", "cormo",  "dum", "king", "pelican", "pes", "prigorie", "sil"]

# Define the parameters for the spectrogram
n_fft = 2048
hop_length = 512
n_mels = 128

SECTION_DURATION = 3.0  # Fixed section duration in seconds

def detect(audio_file):
    # Load and preprocess the audio file
    audio, sr = librosa.load(audio_file)

    # Calculate the number of samples for each section
    section_samples = int(sr * SECTION_DURATION)

    # Split the audio into sections
    sections = [audio[i:i+section_samples] for i in range(0, len(audio), section_samples)]

    results = []

    for section in sections:
        # Preprocess the audio section
        S = librosa.feature.melspectrogram(y=section, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)
        desired_shape = (128, 130)

        print("Section shape: ", S_dB.shape)
        # Pad or truncate the spectrogram to the desired shape
        file_shape = S_dB.shape
        if (file_shape[1] < desired_shape[1] and file_shape[1] > 0.66 * desired_shape[1]):
            pad_amount = [(0, 0), (0, desired_shape[1] - file_shape[1])]
            S_dB = np.pad(S_dB, pad_amount, mode='constant')
        elif file_shape[1] > desired_shape[1]:
            S_dB = S_dB[:, :desired_shape[1]]
        elif file_shape[1] <= 0.66 * desired_shape[1]:
            # Skip processing and move to the next section
            results.append("Recording segment too short.")
            continue

        # Define the threshold for the minimum confidence prediction
        threshold = 0.7

        # Call the predict method of the model to obtain a probability distribution over the possible classes
        predictions = model.predict(np.array([S_dB]))

        # Create an array to store the predicted probabilities for all the classes
        predicted_probabilities = predictions[0]

        # Print the predicted class and its corresponding probability
        if np.max(predicted_probabilities) >= threshold:
            if (bird_species[np.argmax(predicted_probabilities)] == "ana"):
                sp = "Rața sulițar"
            elif (bird_species[np.argmax(predicted_probabilities)] == "chira"):
                sp = "Chira de baltă"
            elif (bird_species[np.argmax(predicted_probabilities)] == "cormo"):
                sp = "Cormoranul mare"
            elif (bird_species[np.argmax(predicted_probabilities)] == "dum"):
                sp = "Dumbrăveanca"
            elif (bird_species[np.argmax(predicted_probabilities)] == "king"):
                sp = "Pescărușul albastru"
            elif (bird_species[np.argmax(predicted_probabilities)] == "pelican"):
                sp = "Pelicanul comul"
            elif (bird_species[np.argmax(predicted_probabilities)] == "pes"):
                sp = "Pescărușul mediteranean"
            elif (bird_species[np.argmax(predicted_probabilities)] == "prigorie"):
                sp = "Prigoria"
            elif (bird_species[np.argmax(predicted_probabilities)] == "sil"):
                sp = "Silvia comună"
            result = (sp, round(np.max(predicted_probabilities) * 100, 2))
        else:
            result = "The model is not confident enough to make a prediction with a minimum confidence of 70%."
        results.append(result)

    return results
