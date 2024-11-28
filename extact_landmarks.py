import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7 )

DATA_DIR = './new_data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            print(f"Error loading image: {os.path.join(DATA_DIR, dir_, img_path)}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            print(f"No hand landmarks found in {os.path.join(DATA_DIR, dir_, img_path)}")
            continue

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                width = max(x_) - min(x_)
                height = max(y_) - min(y_)
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append((x - min(x_)) / width)
                    data_aux.append((y - min(y_)) / height)

            data.append(data_aux)
            labels.append(int(dir_))

f = open('Adata.pickle', 'wb')
metadata = {'classes': 26, 'samples': len(data)}
pickle.dump({'data': data, 'labels': labels, 'metadata': metadata}, f)
f.close()