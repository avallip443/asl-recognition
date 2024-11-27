import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import logging
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load SVM model for ASL prediction
model_dict = pickle.load(open('./model/svm_model.p', 'rb'))
model = model_dict['model']

# Set up webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, max_num_hands=1)

# Dictionary mapping numeric predictions to ASL characters
labels_dict = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e',
    5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
    10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o',
    15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't',
    20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z',
    26: '0', 27: '1', 28: '2', 29: '3', 30: '4',
    31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: 'I love You', 37: 'yes', 38: 'No', 39: 'Hello',
    40: 'Thanks', 41: 'Sorry', 43: 'space'
}

# Variables for tracking predictions and timing
prev_prediction = None
last_detected_character = None
fixed_character = ""
delayCounter = 0
start_time = time.time()

def run():
    """
    Captures video, detects hand landmarks, predicts ASL characters, and displays them on the video.
    """
    global last_detected_character, fixed_character, delayCounter, start_time

    while True:
        data_aux = []  # Auxiliary data for predictions
        x_coords = []  # x-coordinates of landmarks
        y_coords = []  # y-coordinates of landmarks

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape  # Get frame dimensions
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert image to RGB
        results = hands.process(frame_rgb)  # Process frame for landmarks

        if results.multi_hand_landmarks:  # Landmarks detected
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract normalized landmark points
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_coords.append(x)
                    y_coords.append(y)

                # Calculate features for model prediction
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_coords))
                    data_aux.append(y - min(y_coords))

                # Bounding box for the hand
                x1 = int(min(x_coords) * W) - 10
                y1 = int(min(y_coords) * H) - 10
                x2 = int(max(x_coords) * W) - 10
                y2 = int(max(y_coords) * H) - 10

                # Make prediction using the model
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Draw prediction on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                # Timer to check if predicted character is stable for at least 1 second
                current_time = time.time()
                if predicted_character == last_detected_character:  # Prediction stable for 1 sec
                    if (current_time - start_time) >= 1.0:
                        fixed_character = predicted_character

                        if delayCounter == 0:  # Update frame with text
                            logging.info(f"Detected: {fixed_character}")
                            delayCounter = 1
                else:  # Reset timer when another character is detected
                    start_time = current_time
                    last_detected_character = predicted_character
                    delayCounter = 0

        # Show video with prediction
        cv2.imshow('ASL Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        logging.error(f"Error occurred: {e}")
