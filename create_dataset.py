import os
import pickle
import cv2
import mediapipe as mp 
import matplotlib.pyplot as plt

# access kaggle dataset
DATA_DIR = './extracted_folder/asl_alphabet_train/asl_alphabet_train'
OUTPUT_FILE = 'kaggle_data.pickle'

# initialize mediapipe modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
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
    
    for class_dir in class_directories:
        class_path = os.path.join(data_dir, class_dir)
        if not os.path.isdir(class_path):
            print(f"Skipping file: {class_dir}")
            continue 
        
        print(f"Processing class directory: {class_dir}")
        image_files = os.listdir(class_path)
        
        # process image in class dir
        for image_file in image_files: 
            image_path = os.path.join(class_path, image_file)

            try:
                data_points, label, shape = process_image(image_path, class_dir)

                if len(data_points) == 21:  # Mediapipe returns 21 points, each with (x, y)
                    data.append(data_points)
                    labels.append(label)
                else:
                    print(f"Skipping image due to incomplete landmarks: {image_file}")
                    print(f"Incomplete landmarks in {image_file}: Dimensions {shape} Length {len(data_points)}")

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

    save_dataset(output_file, data, labels)
    plt.ioff()  # Turn off interactive mode
    print(f"Dataset saved to {output_file}")


def process_image(image_path, class_label):
    """
    Process image to get hand landmarks and return data points and label.
    
    Args:
        image_path (str): Path to the image.
        class_label (str): Label of the image class.
    Returns:
        tuple: Tuple containing the data points and class label.
    """
    print(f"Processing image: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
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
    else:
        print(f"No hand landmarks detected in {image_path}")
    
    return data_points, class_label, img.shape


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
