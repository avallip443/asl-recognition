import os
import pickle
import cv2
import mediapipe as mp 
import matplotlib.pyplot as plt


# access dataset
DATA_DIR = './new_data'
OUTPUT_FILE = 'new_data.pickle'

# initialize mediapipe modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


def create_dataset(data_dir, output_file):
    """
    Create dataset from images from class directories.
    
    Args:
        data_dir (str): Directory containing subdirectories of class images.
        output_file (str): File path to save dataset as a pickle file.
    """
    print(f"Checking DATA_DIR at: {os.path.abspath(data_dir)}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")
    
    # dnitialize data, label lists
    data = []
    labels = []
    
    plt.ion() # turn on interactive mode for visualization
    
    # iterate through class directories (letters A to Z)
    class_directories = os.listdir(data_dir)
    if not class_directories:
        raise ValueError(f"No class directories found in '{data_dir}'.")
    
    print("Creating dataset...")
    
    # loop through directories and process images
    for class_dir in class_directories:
        class_path = os.path.join(data_dir, class_dir)
        if not os.path.isdir(class_path):
            print(f"Skipping non-directory: {class_dir}")
            continue 
        
        print(f"Processing class directory: {class_dir}")
        image_files = os.listdir(class_path)
        
        # process image in class dir
        for image_file in image_files: 
            image_path = os.path.join(class_path, image_file)

            try:
                process_image(image_path, class_dir, data, labels)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

    save_dataset(output_file, data, labels)
    plt.ioff()  # turn off interactive mode
    print(f"Dataset saved to {output_file}")


def process_image(image_path, class_label, data, labels):
    """
    Process image to get hand landmarks and return data points and label.
    
    Args:
        image_path (str): Path to the image.
        class_label (str): Label of the image class.
        data (list): List to store extracted data points (landmark coordinates).
        labels (list): List to store corresponding labels for data points.
    """
    print(f"Processing image: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # resize image 
    #img = cv2.resize(img, (224, 224))
    
    # convert image to rgb
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb) 
        
    # get hand landmark data
    data_points = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_points = [landmark.x for landmark in hand_landmarks.landmark]
            y_points = [landmark.y for landmark in hand_landmarks.landmark]
            data_points.extend(zip(x_points, y_points))
    
    if len(data_points) == 21:  # 21 landmarks, each with (x, y)
        data.append(data_points)
        labels.append(class_label)
    else:
        print(f"Skipping image due to incomplete landmarks: {image_path}")


def save_dataset(output_file, dataset, labels):
    """
    Save dataset to a pickle file.
    
    Args:
        output_file (str): File path to save the dataset.
        dataset (list): List of data points from images.
        labels (list): List of labels corresponding to the data points.
    """
    print("Saving dataset...")
    with open(output_file, 'wb') as f:
        pickle.dump({'data': dataset, 'labels': labels}, f)


if __name__ == "__main__":
    try:
        create_dataset(DATA_DIR, OUTPUT_FILE)
    except Exception as e:
        print(f"Error occurred: {e}")
        