import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# create classes for each symbol (26 for each letter)
number_of_classes = 26  
dataset_size = 100  

cap = cv2.VideoCapture(0)  # change index based on camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 620)   

# create folders for each class
for i in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(i))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {i}')

    # wait for user input to start collecting for each class
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # flip camera
        
        if not ret:
            print("Failed to capture image.")
            break
        
        cv2.putText(frame, 'Press "1" to start, "q" to quit', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(25)  # 25 millisecond delay
        if key == ord('1'):  # start collecting data 
            break 
        elif key == ord('q'):  # exit program
            cap.release()
            cv2.destroyAllWindows()
            exit() 

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break
        cv2.imshow('frame', frame)
        
        # save captured image
        file_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(file_path, frame)
        print(f'Saved {file_path}')
        counter += 1
        
        if cv2.waitKey(25) == ord('q'):  # exit program 
            cap.release()
            cv2.destroyAllWindows()
            exit() 

cap.release()
cv2.destroyAllWindows()
