import pyaudio
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import write
import librosa
import joblib

# Constants
CHUNK = 2048                # Number of audio samples per chunk
FORMAT = pyaudio.paInt16     # 16-bit audio format
CHANNELS = 1                 # Mono audio
RATE = 16000                 # Sample rate (Hz)
SEGMENT_DURATION = 3         # Duration of each segment in seconds
MAX_PAD_LEN = 220            # Padding length for log-mel spectrogram

model = tf.keras.models.load_model('speech_model.h5')

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

def extract_log_mel_spectrogram(audio, sample_rate, max_pad_len=MAX_PAD_LEN):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    log_mel_spectrogram = librosa.util.fix_length(log_mel_spectrogram, size=MAX_PAD_LEN)
    return log_mel_spectrogram

def extract_features_from_segment(audio_segment, sample_rate=RATE, max_pad_len=MAX_PAD_LEN):
    try:
        audio = audio_segment.astype(np.float32)
        if sample_rate != RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=RATE)

        log_mel_spectrogram = extract_log_mel_spectrogram(audio, RATE, max_pad_len)
        return log_mel_spectrogram
    except Exception as e:
        print(f"Error processing audio segment: {e}")
        return None

label_encoder = joblib.load('label_encoder.pkl')


try:
    print("Listening for audio... Press Ctrl+C to stop.")
    while True:
        frames = []
        
        for _ in range(int(RATE / CHUNK * SEGMENT_DURATION)):
            print("Recording...")
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.int16))
            except Exception as e:
                print(f"Error reading audio data: {e}")

        audio_segment = np.concatenate(frames)
        preprocessed_segment = extract_features_from_segment(audio_segment)
        
        if preprocessed_segment is not None:
            preprocessed_segment = np.expand_dims(preprocessed_segment, axis=-1)
            preprocessed_segment = np.expand_dims(preprocessed_segment, axis=0)
            prediction = model.predict(preprocessed_segment)
            print(prediction)
            predicted_index = np.argmax(prediction, axis=1)
            predicted_emotion = label_encoder.inverse_transform(predicted_index)
            print(f"Predicted Emotion and Intensity: {predicted_emotion}")
        
except KeyboardInterrupt:
    print("Stopped listening")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
