import soundcard as sc
import numpy as np
import wave
import os
import datetime

def test_microphone():
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5

    # Generate a unique filename using the current date and time
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    WAVE_OUTPUT_FILENAME = f"test_recording_{timestamp}.wav"

    try:
        # Get default microphone
        mic = sc.default_microphone()

        print("* Recording audio for 5 seconds")
        recording = mic.record(samplerate=RATE, numframes=RATE*RECORD_SECONDS, channels=CHANNELS)
        print("* Done recording")

        # Normalize the recording to 16-bit range
        recording = np.int16(recording / np.max(np.abs(recording)) * 32767)

        # Save as WAV file
        with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 2 bytes for 'int16' dtype
            wf.setframerate(RATE)
            wf.writeframes(recording.tobytes())

        print(f"* Audio saved as {WAVE_OUTPUT_FILENAME}")

        # Get the full path of the file
        full_path = os.path.abspath(WAVE_OUTPUT_FILENAME)
        print(f"* Full path: {full_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_microphone()
