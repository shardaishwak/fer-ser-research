import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

print("Devices available:", tf.config.list_physical_devices())

model = load_model('facial_model.h5')

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_intensity_map = {
    'ANG_LO': 0, 'ANG_MD': 1, 'ANG_HI': 2,
    'FEA_LO': 3, 'FEA_MD': 4, 'FEA_HI': 5,
    'HAP_LO': 6, 'HAP_MD': 7, 'HAP_HI': 8,
    'SAD_LO': 9, 'SAD_MD': 10, 'SAD_HI': 11,
    'DIS_LO': 12, 'DIS_MD': 13, 'DIS_HI': 14
}
intensity_label_map = {v: k for k, v in emotion_intensity_map.items()}

# Give nice description to the labels: ANG_LO = Angry Low
intensity_label_map = {0: 'Angry Low', 1: 'Angry Medium', 2: 'Angry High',
                       3: 'Fear Low', 4: 'Fear Medium', 5: 'Fear High',
                       6: 'Happy Low', 7: 'Happy Medium', 8: 'Happy High',
                       9: 'Sad Low', 10: 'Sad Medium', 11: 'Sad High',
                       12: 'Disgust Low', 13: 'Disgust Medium', 14: 'Disgust High'}


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        resized_face = cv2.resize(face_img, (48, 48))
        resized_face = resized_face.reshape(1, 48, 48, 1)
        

        frame[0:48, 0:48] = resized_face

        normalized_face = resized_face / 255.0

        output_vector = model.predict(normalized_face)
        predicted_label_index = np.argmax(output_vector, axis=1)[0]


        # show Image rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)


        # Show text
        text = intensity_label_map.get(predicted_label_index)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x + w - text_size[0]
        text_y = y - 10 
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    cv2.imshow('Live Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
