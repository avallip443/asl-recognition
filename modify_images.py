import os
import cv2

INPUT_DIR = './data'  # directory with original images
OUTPUT_DIR = './mod_data'  # directory to save images
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

NUMBER_OF_CLASSES = 26  

def crop_image_to_square(image):
    """
    Crop given image to a square shape (focuses on center area).
    
    Args:
        image (numpy array): Input image as a numpy array.
    Returns:
        numpy array: Cropped square image.
    """
    height, width = image.shape[:2]
    size = min(height, width)
    x_start = (width - size) // 2
    y_start = (height - size) // 2
    return image[y_start:y_start + size, x_start:x_start + size]


def process_images(input_dir, output_dir, target_size=None):
    """
    Process and crop all images in the input directory, save to output directory.
    
    Args:
        input_dir (str): Directory of input images.
        output_dir (str): Directory to save cropped images.
        target_size (tuple): Resize cropped image to size (width, height).
    """
    
    for class_id in range(NUMBER_OF_CLASSES):
        class_input_dir = os.path.join(input_dir, str(class_id))
        class_output_dir = os.path.join(output_dir, str(class_id))
        
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)
            
        if not os.path.exists(class_input_dir):
            print(f"Class directory {class_input_dir} does not exist.")
            continue


        for file in os.listdir(class_input_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(class_input_dir, file)
                output_path = os.path.join(class_output_dir, file)

                # read image
                image = cv2.imread(input_path)
                if image is None:
                    print(f"Failed to load image: {input_path}")
                    continue
                
                # crop iamge to square
                square_image = crop_image_to_square(image)
                
                # resize if target size is provided
                if target_size:
                    square_image = cv2.resize(square_image, target_size)
                
                # save cropped image
                cv2.imwrite(output_path, square_image)
                print(f"Saved cropped image to: {output_path}")
    '''
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)
            
            # Read image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to load image: {input_path}")
                continue
            
            # Crop to square
            square_image = crop_image_to_square(image)
            
            # Resize if target size is provided
            if target_size:
                square_image = cv2.resize(square_image, target_size)
            
            # Save the cropped image
            cv2.imwrite(output_path, square_image)
            print(f"Saved cropped image to: {output_path}")
    '''
     
# crop all images and resize to 28x28
process_images(INPUT_DIR, OUTPUT_DIR, target_size=(512, 512))
