import os
import cv2

# directory to save the dataset
DATA_DIR = './new_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# number of classes (eg. A, B) and number of images per class
NUMBER_OF_CLASSES = 3  
DATASET_SIZE  = 500  

# set up webcam 
cap = cv2.VideoCapture(0)  # change index if multiple cameras are connected
# image dimensions are 512x512 pixels
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)   

# check if webcam opened
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# create folders for each class
for class_id in range(NUMBER_OF_CLASSES):
    class_dir = os.path.join(DATA_DIR, str(class_id))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Creating folder for class {class_id}: {class_dir}')
    
# collect data for each class
for class_id in range(NUMBER_OF_CLASSES):
    print(f'Collecting data for class {class_id}')
    class_dir = os.path.join(DATA_DIR, str(class_id))
    
    # wait for user input to start collecting for each class
    while True:
        ret, frame = cap.read()        
        if not ret:
            print("Failed to capture image.")
            break

        # frame = cv2.flip(frame, 1)  # flip camera        
        cv2.putText(frame, f'Collecting Class {class_id}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.putText(frame, 'Press "1" to start or "q" to quit', (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(25)  # 25 millisecond delay
        if key == ord('1'):  # start collecting data 
            break 
        elif key == ord('q'):  # exit program
            print("Exiting program...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # collect images for current class
    counter = 0
    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break
        
        cv2.putText(frame, f'Class {class_id} - Image {counter + 1}/{DATASET_SIZE}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        
        # save captured image
        file_path = os.path.join(class_dir, f'{counter}.jpg')
        
        try:
            cv2.imwrite(file_path, frame)
            print(f'Successfully saved: {file_path}')
            counter += 1
        except Exception as e:
            print(f"Error saving image {file_path}: {e}")
        
        # user quits programs midway
        if cv2.waitKey(25) == ord('q'):  
            print("Exiting program...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

print("Data collection completed successfully for all classes.")

# release webcam and close cv windows
cap.release()
cv2.destroyAllWindows()
