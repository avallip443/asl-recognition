import os
import pickle
import cv2
import mediapipe as mp 
import matplotlib.pyplot as plt

print("Visualizing images...")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
print(f"Checking DATA_DIR at: {os.path.abspath(DATA_DIR)}")

if not os.path.exists(DATA_DIR):
    print(f"Directory {DATA_DIR} does not exist.")
else:     
    dir_contents = os.listdir(DATA_DIR)
   
    for class_dir in dir_contents:
        class_path = os.path.join(DATA_DIR, class_dir)
        print(f"Class directory: {class_dir}")
        
        img_files = os.listdir(class_path)

        for img_path in img_files[:1]:
            img_full_path = os.path.join(class_path, img_path)
            img = cv2.imread(img_full_path)
            
            if img is None:
                print(f"Failed to load image from {img_full_path}")
                continue
            else:
                print(f"Successfully loaded {img_full_path}")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        
                        mp_drawing.draw_landmarks(
                            img_rgb,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                        
                plt.figure()  
                plt.imshow(img_rgb)
                plt.axis('off')  # hide axis
                plt.draw()  
                plt.pause(1)  # 1 second pause
                #plt.clf()
                
    plt.show()  
    plt.ioff() 
                        
