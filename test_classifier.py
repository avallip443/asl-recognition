import cv2
import mediapipe as mp
import numpy as np
import pickle

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.3)

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Labels for gestures
labels_dict = {0: 'A', 1: 'B', 2: 'C'}

# Function to extract and normalize the landmark data for prediction
def extract_landmark_data(hand_landmarks, H, W):
    data_aux, x_coords, y_coords = [], [], []
    for landmark in hand_landmarks.landmark:
        x, y = landmark.x, landmark.y
        x_coords.append(x)
        y_coords.append(y)
    
    min_x, min_y = min(x_coords), min(y_coords)
    max_x, max_y = max(x_coords), max(y_coords)

    for landmark in hand_landmarks.landmark:
        normalized_x = (landmark.x - min_x) / (max_x - min_x)
        normalized_y = (landmark.y - min_y) / (max_y - min_y)
        
        data_aux.append(normalized_x)
        data_aux.append(normalized_y)

    return data_aux, x_coords, y_coords

def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        H, W, Z = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame for landmarks
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )

                data_aux, x_, y_ = extract_landmark_data(hand_landmarks, H, W)
                data_aux = np.asarray(data_aux) / np.max(data_aux)
                data_aux = data_aux.flatten()

                if len(data_aux) == 42:
                    x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                    x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
                    
                    # Make prediction using the model
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                else:
                    print("Unexpected input size for the model.")
        else:
            cv2.putText(frame, "No hands detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame.")
            continue

        # Yield the frame as part of the MJPEG stream
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
