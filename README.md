# ASL Recognition System

## Description
This project is an ASL (American Sign Language) recognition system that processes live video input to identify and display ASL signs in real time. Using computer vision and machine learning, the program accurately recognizes all 26 letter gestures and provides real=time predictions. 

## Table of Contents
- [Training Model](#training-model)
    - [Data Collection](#data-collection)
    - [Dataset Creation](#dataset-creation)
    - [Training SVM Model](#training-svm-model)
- [Installation](#installation)
- [Dependencies](#dependencies)


## Training Model
This section describes collecting data, extracting meaningingful features, and using these to train a support vector machine (SVM) model capable of making reliable ASL sign predictions. 

## Set Up
Before running the scripts for training the model, it is recommended to use a virtual environment.

1. Install `virtualenv ` (if not already installed)
```bash
pip install virtualenv
```
2. Create a virtual environment
```bash
python -m venv venv
```
3. Active the virtual environment
- Windows:
```bash
.\venv\Scripts\activate
```
- macOS/Linux:
```bash
source venv/bin/activate
```
4. Install project dependencies
```bash
pip install -r requirements.txt
```


### Data Collection
The script `collect_data.py` collects and organizes images required to train the model. Using a webcam, it takes 300 images of 244x244 pixels for each ASL gesture and saves them in labeled directories corresponding to their gestures. For example, images for the first gesture ('A') would be stored in `./sign_images/0`. 

#### Steps in Data Collection
1. Launch the script by entering:
```bash
python collect_data.py
```
2. Press '1' to start capturing images for a gesture.
3. Press 'q' t quit the program.


### Dataset Creation
The script `create_dataset.py` processes the captured images into meaningful features for training the model. Using the `Mediapipe` library, it extracts hand landmarks (the x and y coordinates of 21 points on the hand) from each image to be used as input data for the SVM model. Each gesture is assigned a unique numberic label corresponding to its directoriy (eg. 'A' = 0, 'B' = 1).

#### Steps in Dataset Creation
1. Launch the script by entering:
```bash
python create_dataset.py
```
2. Upon completion, an output file called `data.pickle' will be creating containing the processed dataset.


### Training SVM Model


## Installation


## Dependencies


## Data Collection


6. Start the UI
```bash
python app.py
```
6. Navigate to address on browser 

    Look for
    ```bash 
    Running on http://127.0.0.XXXXX 
    ```
    in terminal and navigate to that link on your browser
   
## Dataset Creation

### Set up
It is assumed that the virtual environment has been set up and the images are in the `./data` directory before running the script.

1. Run the script
```bash
python create_dataset.py
```
2. Output
- The processed dataset will be saves as `.pickle` in the rot directory.

### Important Notes
- Ensure that the images in the `./data` directory are of good quality and contain visible hands for best results.

## Dependencies
The following libraries are required to run the project:
- `flask`
- `mediapipe`
- `matplotlib`
- `numpy`
- `opencv-python`
- `os`
- `pickle`
- `scikit-learn`

## Installation
Install the required libraries with the following command:

```bash
pip install flask opencv-python mediapipe matplotlib numpy scikit-learn
```
