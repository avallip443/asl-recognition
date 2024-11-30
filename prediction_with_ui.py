import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import threading
import time
import logging
import warnings

# suppress specific warnings :)
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")

# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load svm model for asl prediction
model_dict = pickle.load(open('./model/svm_model.p', 'rb'))
model = model_dict['model']

# set up webcam
cap = cv2.VideoCapture(0)

# initialize mediapipe modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, max_num_hands=1)

# dictionary mapping numeric predications to asl characters
labels_dict = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 
    5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 
    10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 
    15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 
    20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'
}  

# window for output
root = tk.Tk()
root.title("ASL Prediction")

# add text field to display signs
text_field = tk.Text(root, height=2, width=40, font=("Helvetica", 16))
text_field.pack()


def clear_text():
    """
    Clears the text field in the tkinter window.
    """
    text_field.delete('1.0', tk.END)  
    logging.info('Text cleared.')


# add button to clear text
clear_button = tk.Button(root, text="Clear Text", command=clear_text)
clear_button.pack()

# variables for tracking predications, timing
prev_prediction = None
word_count = 0  # track words written
last_detected_character = None
fixed_character = ""
delayCounter = 0
start_time = time.time()


def update_text_field(text):
    """
    Updates the tkinter text field with the predicted character/word.
    """    
    if text == 'space':  # add space
        text_field.insert(tk.END, ' ')  
    else:  # add character
        text_field.insert(tk.END, text + '')  
    logging.info(f'Word added: {text if text != "space" else "space (represented as space)"}')


def run():
    """
    Captures video, detects hand landmarks, predicts ASL characters, and updates the tkinter text field.
    """
    global last_detected_character, fixed_character, delayCounter, start_time

    while True:
        data_aux = []  # auxiliary data for predications
        x_coords = []  # x-coords of landmarks
        y_coords = []  # y-coords of landmarks

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape  # get frame dimensions
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert image to rgb
        results = hands.process(frame_rgb)  # process frame for landmarks

        if results.multi_hand_landmarks:  # landmarks detected
            for hand_landmarks in results.multi_hand_landmarks:
                # draw landmarks on frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            #for hand_landmarks in results.multi_hand_landmarks:
                # extract normalizes landmark points
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_coords.append(x)
                    y_coords.append(y)

                # calculate features for model prediction
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_coords))
                    data_aux.append(y - min(y_coords))

                # box for hand
                x1 = int(min(x_coords) * W) - 10
                y1 = int(min(y_coords) * H) - 10
                x2 = int(max(x_coords) * W) - 10
                y2 = int(max(y_coords) * H) - 10
                
                # make predication using model
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # draw predication on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                # timer to check if predicated character is same for at least 1 second duration
                current_time = time.time()
                if predicted_character == last_detected_character: # predication stable for 1 sec
                    if (current_time - start_time) >= 1.0:  
                        fixed_character = predicted_character
                        
                        if delayCounter == 0:  # update text field 
                            update_text_field(fixed_character)
                            delayCounter = 1
                else:  # reset timer when another character detected
                    start_time = current_time
                    last_detected_character = predicted_character
                    delayCounter = 0  

        # show video with prediction
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def exit_app():
    """
    Exits all applications.
    """
    global cap
    logging.info('Exiting application...')
    
    if cap.isOpened():
        cap.release()  
        
    cv2.destroyAllWindows()  
    root.quit()  
    root.destroy()  

# add exit button
exit_button = tk.Button(root, text="Exit", command=exit_app)
exit_button.pack()

# video separate from tkinter
threading.Thread(target=run, daemon=True).start()

root.mainloop()
