import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


# access dataset
DATA_DIR = './sign_images'
OUTPUT_FILE = 'Adata.pickle'

# initialize mediapipe modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)

# check if direcotry exists
print(f"Checking DATA_DIR at: {os.path.abspath(DATA_DIR)}")
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory '{DATA_DIR}' does not exist.")
    
# initialize data, label lists
data = []
labels = []

plt.ion() # turn on interactive mode for visualization

# iterate through class directories (letters A to Z)
class_directories = os.listdir(DATA_DIR)
if not class_directories:
    raise ValueError(f"No class directories found in '{DATA_DIR}'.")
    
print("Creating dataset...")

for class_dir in class_directories:
    class_path = os.path.join(DATA_DIR, class_dir)
    
    if not os.path.isdir(class_path):
        print(f"Skipping non-directory: {class_dir}")
        continue
    
    print(f"Processing class directory: {class_dir}")
    
    image_files = os.listdir(class_path) 
    
    for image_file in image_files:
        data_aux = []
        x_coords = []
        y_coords = []

        image_path = os.path.join(class_path, image_file)
        
        # laod image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {os.path.join(DATA_DIR, class_dir, image_file)}")
            continue

        # convert image to rgb
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # process image and detect landmarks 
        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            print(f"No hand landmarks found in {os.path.join(DATA_DIR, class_dir, image_file)}")
            continue
        
        # get hand landmark data and normalize
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_coords.append(x)
                    y_coords.append(y)

                # dimensions of hand in iamge
                width = max(x_coords) - min(x_coords)
                height = max(y_coords) - min(y_coords)
                
                # normalize coordinates of landmark
                for i in range(len(hand_landmarks.landmark)):
                    x_normalized = (hand_landmarks.landmark[i].x - min(x_coords)) / width
                    y_normalized = (hand_landmarks.landmark[i].y - min(y_coords)) / height
                    data_aux.append(x_normalized)
                    data_aux.append(y_normalized)

            # append data and corresponding label
            data.append(data_aux)
            labels.append(int(class_dir))

plt.ioff()  # turn off interactive mode

# save the dataset as a pickle file
with open(OUTPUT_FILE, 'wb') as f:
    metadata = {'classes': 26, 'samples': len(data)}  # assuming 26 classes (A-Z)
    pickle.dump({'data': data, 'labels': labels, 'metadata': metadata}, f)

print(f"Dataset saved as {OUTPUT_FILE}")