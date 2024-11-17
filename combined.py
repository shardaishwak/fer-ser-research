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

latest_audio_prediction = None
latest_video_prediction = None
speech_labels = None 

audio_lock = threading.Lock()
video_lock = threading.Lock()

def audio_processing():
    global latest_audio_prediction
    global speech_labels

    # Load the speech model
    speech_model = tf.keras.models.load_model('speech_model.h5')
    label_encoder = joblib.load('label_encoder.pkl')
    speech_labels = label_encoder.classes_  # Assign speech_labels here

    # Map speech labels to indices
    speech_label_to_index = {label: idx for idx, label in enumerate(speech_labels)}

    # Constants
    CHUNK = 2048
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    SEGMENT_DURATION = 3
    MAX_PAD_LEN = 220

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    def extract_log_mel_spectrogram(audio_data, sample_rate, max_pad_len=MAX_PAD_LEN):
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        log_mel_spectrogram = librosa.util.fix_length(log_mel_spectrogram, size=max_pad_len)
        return log_mel_spectrogram

    try:
        while True:
            frames = []
            # Capture audio segment
            for _ in range(int(RATE / CHUNK * SEGMENT_DURATION)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.int16))
            audio_segment = np.concatenate(frames)

            # Voice Activity Detection (VAD): To make sure we process only when speech is detected
            rms = np.sqrt(np.mean(audio_segment.astype(np.float32)**2))
            ENERGY_THRESHOLD = 500
            if rms < ENERGY_THRESHOLD:
                with audio_lock:
                    latest_audio_prediction = None
                continue

            audio_data = audio_segment.astype(np.float32)
            if RATE != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=RATE, target_sr=16000)

            log_mel_spectrogram = extract_log_mel_spectrogram(audio_data, 16000, MAX_PAD_LEN)
            preprocessed_segment = np.expand_dims(log_mel_spectrogram, axis=-1)
            preprocessed_segment = np.expand_dims(preprocessed_segment, axis=0)

            logger.debug(f"Preprocessed segment shape: {preprocessed_segment.shape}")
            prediction = speech_model.predict(preprocessed_segment)
            with audio_lock:
                latest_audio_prediction = prediction

    except Exception as e:
        logger.error(f"An error occurred in audio_processing: {e}")
        e.with_traceback()
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

def main():
    global latest_video_prediction
    global speech_labels

    audio_thread = threading.Thread(target=audio_processing)
    audio_thread.start()

    # Wait for speech_labels to be initialized
    while speech_labels is None:
        time.sleep(0.1)

    facial_model = load_model('facial_model.h5')

    facial_labels = [
        'ANG_HI', 'ANG_MD', 'ANG_LO',
        'FEA_HI', 'FEA_MD', 'FEA_LO',
        'HAP_HI', 'HAP_MD', 'HAP_LO',
        'SAD_HI', 'SAD_MD', 'SAD_LO',
        'DIS_HI', 'DIS_MD', 'DIS_LO'
    ]

    facial_label_to_common = {
        'ANG_HI': 'angry-high',
        'ANG_MD': 'angry-medium',
        'ANG_LO': 'angry-low',
        'FEA_HI': 'fear-high',
        'FEA_MD': 'fear-medium',
        'FEA_LO': 'fear-low',
        'HAP_HI': 'happy-high',
        'HAP_MD': 'happy-medium',
        'HAP_LO': 'happy-low',
        'SAD_HI': 'sad-high',
        'SAD_MD': 'sad-medium',
        'SAD_LO': 'sad-low',
        'DIS_HI': 'disgust-high',
        'DIS_MD': 'disgust-medium',
        'DIS_LO': 'disgust-low'
    }

    # Speech labels: this will be used as common mapping labels
    common_labels = [
        'angry-low', 'angry-medium', 'angry-high',
        'fear-low', 'fear-medium', 'fear-high',
        'happy-low', 'happy-medium', 'happy-high',
        'sad-low', 'sad-medium', 'sad-high',
        'disgust-low', 'disgust-medium', 'disgust-high'
    ]

    # Create mappings for facial model to common labels
    facial_label_to_index = {label: idx for idx, label in enumerate(facial_labels)}
    common_label_to_facial_index = {}
    for facial_label, idx in facial_label_to_index.items():
        common_label = facial_label_to_common[facial_label]
        common_label_to_facial_index[common_label] = idx

    # Create mappings for speech model to common labels
    speech_label_to_index = {label: idx for idx, label in enumerate(speech_labels)}
    common_label_to_speech_index = {label: speech_label_to_index.get(label, -1) for label in common_labels}

    audio_weight = 0.3
    video_weight = 0.7

    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

            with video_lock:
                latest_video_prediction = None

            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                resized_face = cv2.resize(face_img, (48, 48))
                resized_face = resized_face.reshape(1, 48, 48, 1)

                normalized_face = resized_face / 255.0

                output_vector = facial_model.predict(normalized_face)

                # Update latest_video_prediction
                with video_lock:
                    latest_video_prediction = output_vector

                # Map facial model predictions to common labels
                facial_probs = output_vector[0]
                facial_common_probs = [0] * len(common_labels)
                for common_label in common_labels:
                    idx = common_label_to_facial_index[common_label]
                    prob = facial_probs[idx]
                    facial_common_probs[common_labels.index(common_label)] = prob

                # Get predicted label for display
                predicted_index = np.argmax(facial_common_probs)
                text = common_labels[predicted_index]

                # Draw rectangle and text on frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                )

            # Combine predictions
            with audio_lock:
                audio_pred = latest_audio_prediction
            with video_lock:
                video_pred = latest_video_prediction

            if video_pred is not None:
                # Map facial model predictions to common labels
                facial_probs = video_pred[0]
                facial_common_probs = [0] * len(common_labels)
                for common_label in common_labels:
                    idx = common_label_to_facial_index[common_label]
                    prob = facial_probs[idx]
                    facial_common_probs[common_labels.index(common_label)] = prob
                facial_probs_norm = np.array(facial_common_probs)
                facial_probs_norm /= np.sum(facial_probs_norm)
            else:
                facial_probs_norm = None

            if audio_pred is not None:
                # Map speech model predictions to common labels
                speech_probs = audio_pred[0]
                speech_common_probs = [0] * len(common_labels)
                for common_label in common_labels:
                    idx = common_label_to_speech_index[common_label]
                    if idx != -1:
                        prob = speech_probs[idx]
                        speech_common_probs[common_labels.index(common_label)] = prob
                speech_probs_norm = np.array(speech_common_probs)
                if np.sum(speech_probs_norm) > 0:
                    speech_probs_norm /= np.sum(speech_probs_norm)
                else:
                    speech_probs_norm = None
            else:
                speech_probs_norm = None

            if facial_probs_norm is not None and speech_probs_norm is not None:
                # Combine the probabilities (Weighted Averaging)
                combined_probs = audio_weight * speech_probs_norm + video_weight * facial_probs_norm
            elif facial_probs_norm is not None:
                combined_probs = facial_probs_norm
            elif speech_probs_norm is not None:
                combined_probs = speech_probs_norm
            else:
                print("Waiting for predictions...")
                cv2.imshow('Emotion Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue 

            # Get the predicted emotion
            predicted_index = np.argmax(combined_probs)
            combined_emotion = common_labels[predicted_index]
            print(f"Combined predicted emotion and intensity: {combined_emotion}")

            # Display the combined emotion on the frame
            cv2.putText(
                frame, f"Combined: {combined_emotion}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
            )

            cv2.imshow('Emotion Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt received. Exiting...")
    except Exception as e:
        logger.error(f"An error occurred in main: {e}")
        e.with_traceback()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        audio_thread.join()

if __name__ == "__main__":
    main()
