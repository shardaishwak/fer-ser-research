import cv2
import pyaudio
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import time
import librosa
import joblib
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Shared variables for processed audio data
processed_audio_data = None
processed_audio_timestamp = None
audio_data_lock = threading.Lock()

def audio_processing():
    global processed_audio_data
    global processed_audio_timestamp

    # Constants
    CHUNK = 1024  # Smaller chunk size for more frequent processing
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    MAX_PAD_LEN = 220
    RECORD_SECONDS = 3  # Duration to accumulate audio

    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    audio_buffer = np.array([], dtype=np.int16)

    try:
        while True:
            # Read a small chunk of data
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            audio_buffer = np.concatenate((audio_buffer, audio_chunk))

            # If we have enough data, process it
            if len(audio_buffer) >= RATE * RECORD_SECONDS:
                # Take the last RECORD_SECONDS seconds of data
                audio_segment = audio_buffer[-RATE * RECORD_SECONDS:]

                # Voice Activity Detection (VAD)
                rms = np.sqrt(np.mean(audio_segment.astype(np.float32)**2))
                ENERGY_THRESHOLD = 500  # Adjust as needed
                speech_detected = rms > ENERGY_THRESHOLD

                if speech_detected:
                    audio_data = audio_segment.astype(np.float32)

                    # Resample if needed
                    if RATE != 16000:
                        audio_data = librosa.resample(audio_data, orig_sr=RATE, target_sr=16000)

                    # Extract log-mel spectrogram
                    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=16000)
                    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
                    log_mel_spectrogram = librosa.util.fix_length(log_mel_spectrogram, size=MAX_PAD_LEN)

                    # Prepare audio input for the model
                    audio_input = np.expand_dims(log_mel_spectrogram, axis=-1)
                    audio_input = np.expand_dims(audio_input, axis=0)

                    # Update shared variables with locks
                    with audio_data_lock:
                        processed_audio_data = audio_input
                        processed_audio_timestamp = time.time()
                else:
                    # Reset if no speech detected
                    with audio_data_lock:
                        processed_audio_data = None
                        processed_audio_timestamp = None

                # Remove old data
                audio_buffer = audio_buffer[-RATE * RECORD_SECONDS:]
    except Exception as e:
        logger.error(f"An error occurred in audio_processing: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()

def main():
    global processed_audio_data
    global processed_audio_timestamp
    global audio_data_lock

    # Load the combined model and label encoder
    combined_model = tf.keras.models.load_model('best_combined_model.h5')
    combined_label_encoder = joblib.load('combined_label_encoder.pkl')
    labels = combined_label_encoder.classes_

    # Start the audio processing thread
    audio_thread = threading.Thread(target=audio_processing)
    audio_thread.start()

    # Video capture
    cap = cv2.VideoCapture(0)

    # Load face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Variables for processed video data
    processed_video_data = None
    processed_video_timestamp = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                # Process the first detected face
                (x, y, w, h) = faces[0]
                face_img = gray[y:y + h, x:x + w]
                resized_face = cv2.resize(face_img, (48, 48))
                normalized_face = resized_face / 255.0
                image_input = np.expand_dims(normalized_face, axis=-1)
                image_input = np.expand_dims(image_input, axis=0)

                # Update processed video data
                processed_video_data = image_input
                processed_video_timestamp = time.time()

                # Draw rectangle on the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                processed_video_data = None
                processed_video_timestamp = None

            # Retrieve processed audio data
            with audio_data_lock:
                audio_input = processed_audio_data
                audio_timestamp = processed_audio_timestamp

            # Check if both audio and video data are available and synchronized
            if audio_input is not None and processed_video_data is not None:
                time_diff = abs(processed_video_timestamp - audio_timestamp)
                if time_diff <= 1.0:
                    # Use the combined model
                    prediction = combined_model.predict([audio_input, processed_video_data])
                    predicted_index = np.argmax(prediction)
                    predicted_label = combined_label_encoder.inverse_transform([predicted_index])[0]
                    model_used = 'Combined Model'
                else:
                    predicted_label = 'Processing...'
                    model_used = 'Waiting for synchronized data'
            elif processed_video_data is not None:
                predicted_label = 'Waiting for audio data...'
                model_used = 'Facial Input Detected'
            else:
                predicted_label = 'No face detected'
                model_used = ''

            # Display the result
            cv2.putText(
                frame, f"{predicted_label} ({model_used})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
            )

            cv2.imshow('Emotion Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.05)

    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt received. Exiting...")
    except Exception as e:
        logger.error(f"An error occurred in main: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        audio_thread.join()

if __name__ == "__main__":
    main()
