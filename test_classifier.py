import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf


# Load the trained model
model = load_model('./new_model.h5')

# initalize webcam
cap = cv2.VideoCapture(0)  # index for camaera

# initalize mediapipe modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# labels for gestures
labels_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11: 'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}


# Function to extract and normalize landmark data
def extract_landmark_data(hand_landmarks):
    data_aux = []
    for landmark in hand_landmarks.landmark:
        # Extract normalized x, y coordinates
        data_aux.append([landmark.x, landmark.y])
    
    # Convert to numpy array
    data_aux = np.array(data_aux)

    # Normalize: Divide by max value (consistent with training)
    data_aux /= np.max(data_aux)
    
    return data_aux


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # dimensions of frame
    H, W, _ = frame.shape
    
    # convert frame to rgb
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # process frame for landmarks
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # draw landmarks for each hand
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())    

            # get landmark data for predication
            data_aux = extract_landmark_data(hand_landmarks)
            
            # make prediction if data_aux has the correct features
            if data_aux.shape == (21, 2):
                data_aux = np.expand_dims(data_aux, axis=0)  # Add batch dimension
                
                # Make prediction
                probabilities = tf.nn.softmax(model.predict(data_aux, verbose=0)).numpy()
                confidence = np.max(probabilities)
                predicted_class = np.argmax(probabilities)

                # Check prediction confidence
                predicted_character = labels_dict.get(predicted_class, "Unknown")

                print(f"Predicted: {predicted_character} with confidence {confidence:.2f}")
                
                if confidence > 0.7:  # Confidence threshold
                    # Draw bounding box and label
                    min_x = min([landmark.x for landmark in hand_landmarks.landmark])
                    max_x = max([landmark.x for landmark in hand_landmarks.landmark])
                    min_y = min([landmark.y for landmark in hand_landmarks.landmark])
                    max_y = max([landmark.y for landmark in hand_landmarks.landmark])
                    
                    x1, y1 = int(min_x * W), int(min_y * H)
                    x2, y2 = int(max_x * W), int(max_y * H)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    print("Low confidence. No prediction made.")

            else:
                print("Unexpected input size for the model.")

    else:
        # if no hands are detected, display message
        cv2.putText(frame, "No hands detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the processed frame
    cv2.imshow('Hand Detection', frame)
    cv2.waitKey(1)  
    
    # press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()