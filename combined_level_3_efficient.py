import cv2
import pyaudio
import numpy as np
import tensorflow as tf
import time
import librosa
import joblib
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

def main():
    # Load models
    combined_model = tf.keras.models.load_model('best_combined_model.h5')
    speech_model = tf.keras.models.load_model('speech_model.h5')
    facial_model = tf.keras.models.load_model('facial_model.h5')

    # Load label encoders
    combined_label_encoder = joblib.load('combined_label_encoder.pkl')
    speech_label_encoder = joblib.load('speech_label_encoder.pkl')
    facial_label_encoder = joblib.load('facial_label_encoder.pkl')

    combined_labels = combined_label_encoder.classes_
    speech_labels = speech_label_encoder.classes_
    facial_labels = facial_label_encoder.classes_

    # Video capture
    cap = cv2.VideoCapture(0)

    # Audio capture settings
    CHUNK = 2048
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    MAX_PAD_LEN = 220  # Should match MAX_PAD_LEN used during training
    RECORD_SECONDS = 3  # Duration to record audio

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    # Load face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Voice Activity Detection parameters
    ENERGY_THRESHOLD = 500  # Adjust this threshold based on your environment

    try:
        while True:
            # Capture audio segment
            print("Recording audio...")
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.int16))
            audio_segment = np.concatenate(frames)

            # Voice Activity Detection (VAD)
            rms = np.sqrt(np.mean(audio_segment.astype(np.float32)**2))
            speech_detected = rms > ENERGY_THRESHOLD
            print(f"Audio RMS: {rms}, Speech detected: {speech_detected}")

            # Preprocess audio if speech is detected
            if speech_detected:
                audio_data = audio_segment.astype(np.float32)
                if RATE != 16000:
                    audio_data = librosa.resample(audio_data, orig_sr=RATE, target_sr=16000)
                # Extract log-mel spectrogram
                mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=16000)
                log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
                log_mel_spectrogram = librosa.util.fix_length(log_mel_spectrogram, size=MAX_PAD_LEN)
                # Shape: (128, MAX_PAD_LEN)
                audio_input = np.expand_dims(log_mel_spectrogram, axis=-1)
                audio_input = np.expand_dims(audio_input, axis=0)  # Shape: (1, 128, 220, 1)
            else:
                audio_input = None

            # Capture video frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            # Preprocess image if face is detected
            if len(faces) > 0:
                # Assuming we take the first detected face
                (x, y, w, h) = faces[0]
                face_img = gray[y:y + h, x:x + w]
                resized_face = cv2.resize(face_img, (48, 48))
                normalized_face = resized_face / 255.0
                image_input = np.expand_dims(normalized_face, axis=-1)
                image_input = np.expand_dims(image_input, axis=0)  # Shape: (1, 48, 48, 1)
            else:
                image_input = None

            # Determine which model to use
            if speech_detected and image_input is not None:
                # Both audio and face detected, use combined model
                prediction = combined_model.predict([audio_input, image_input])
                predicted_index = np.argmax(prediction)
                predicted_label = combined_label_encoder.inverse_transform([predicted_index])[0]
                model_used = 'Combined Model'
            elif speech_detected:
                # Only audio detected, use speech model
                prediction = speech_model.predict(audio_input)
                predicted_index = np.argmax(prediction)
                predicted_label = speech_label_encoder.inverse_transform([predicted_index])[0]
                model_used = 'Speech Model'
            elif image_input is not None:
                # Only face detected, use facial model
                prediction = facial_model.predict(image_input)
                predicted_index = np.argmax(prediction)
                predicted_label = facial_label_encoder.inverse_transform([predicted_index])[0]
                model_used = 'Facial Model'
            else:
                # Neither audio nor face detected
                print("No speech or face detected.")
                cv2.imshow('Emotion Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            print(f"Predicted emotion: {predicted_label} (Model used: {model_used})")

            # Display the result on the video frame
            if image_input is not None:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    frame, f"{predicted_label} ({model_used})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
                )
            else:
                cv2.putText(
                    frame, f"{predicted_label} ({model_used})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
                )

            cv2.imshow('Emotion Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Sleep briefly to allow the system to process other events
            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt received. Exiting...")
    except Exception as e:
        logger.error(f"An error occurred in main: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()
