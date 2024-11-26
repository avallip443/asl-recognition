import cv2
import mediapipe as mp
import numpy as np
import pickle

# load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# initalize webcam
cap = cv2.VideoCapture(0)  # index for camaera

# initalize mediapipe modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# labels for gestures
labels_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11: 'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}


# Function to extract and normalize the landmark data for prediction
def extract_landmark_data(hand_landmarks, H, W):
    """
    Extracts and normalizes the hand landmarks for prediction.

    Args:
        hand_landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): 
            A list of landmarks representing the hand detected by MediaPipe.
        H (int): The height of the frame (image).
        W (int): The width of the frame (image).

    Returns:
        tuple: A tuple containing:
            - data_aux (list): List of normalized and scaled landmark coordinates (x, y) for prediction.
            - x_coords (list): List of x coordinates of the landmarks.
            - y_coords (list): List of y coordinates of the landmarks.
    """
    data_aux, x_coords, y_coords = [], [], []
    for landmark in hand_landmarks.landmark:
        # extract x, y coordinates for each landmark
        x, y = landmark.x, landmark.y
        x_coords.append(x)
        y_coords.append(y)
    
    min_x, min_y = min(x_coords), min(y_coords)
    max_x, max_y = max(x_coords), max(y_coords)

    # normalize coordinates to fit in screen
    for landmark in hand_landmarks.landmark:
        # Normalize to [0, 1] range
        normalized_x = (landmark.x - min_x) / (max_x - min_x)
        normalized_y = (landmark.y - min_y) / (max_y - min_y)
        
        data_aux.append(normalized_x)
        data_aux.append(normalized_y)

    return data_aux, x_coords, y_coords


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # dimensions of frame
    H, W, Z = frame.shape
    
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
            data_aux, x_, y_ = extract_landmark_data(hand_landmarks, H, W)
            
            # normalize the data (like in training)
            data_aux = np.asarray(data_aux) / np.max(data_aux)
            
            # flatten data
            data_aux = data_aux.flatten()

            # make prediction if data_aux has the correct features
            if len(data_aux) == 42:  # 21 landmarks * 2 coordinates (x, y)
                #print(f"Feature Data (data_aux): {data_aux}")

                # define box for the detected hand
                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
                
                # make predication using model
                prediction = model.predict([np.asarray(data_aux)])
                #print(f"Model Prediction (raw output): {prediction}")

                # map predication to hand sign
                predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                #print(f"Predicted Character: {predicted_character}")
                
                # Draw rectangle around detected hand and display predicted character
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
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