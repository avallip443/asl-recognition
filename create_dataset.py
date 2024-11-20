import os
import pickle
import cv2
import mediapipe as mp 
import matplotlib.pyplot as plt


print("Creating dataset...")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
print(f"Checking DATA_DIR at: {os.path.abspath(DATA_DIR)}")

if not os.path.exists(DATA_DIR):
    print(f"Directory {DATA_DIR} does not exist.")
else:     
    plt.ion()
    
    data = []
    labels = []
    
    dir_contents = os.listdir(DATA_DIR)
   
    for class_dir in dir_contents:
        class_path = os.path.join(DATA_DIR, class_dir)
        print(f"Class directory: {class_dir}")
        
        # check contents of the class directory
        img_files = os.listdir(class_path)

        for img_path in img_files: 
            print(f"Processing image: {img_path}")
            img_full_path = os.path.join(class_path, img_path)
            img = cv2.imread(img_full_path)
            
            if img is None:
                print(f"Failed to load image from {img_full_path}")
                continue
            else:
                print(f"Successfully loaded {img_full_path}")
                
                data_aux = []
                x_points = []
                y_points = []
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                
                # draw hand landmarks if detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            
                            x_points.append(x)
                            y_points.append(y)
                        data_aux.append((x_points, y_points))
                        
                data.append(data_aux)
                labels.append(class_dir)
    plt.ioff() 
    f = open('data.pickle', 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()
