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

### Set Up
Before running the scripts for training the model, it is recommended to use a virtual environment.

1. Install `virtualenv ` (if not already installed):
```bash
pip install virtualenv
```
2. Create a virtual environment:
```bash
python -m venv venv
```
3. Active the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- macOS/Linux:
```bash
source venv/bin/activate
```
4. Install project dependencies:
```bash
pip install -r model-requirements.txt
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
The script `train_svm_classifier.py` trains an SVM (Support Vector Machine) model to classify ASL gestures. It uses the processed dataset to associate the extracted hand landmarks with their corresponding ASL labels so the model can recognize and predict gestures. The trained model can then be used in the real-time ASL recognition system.

#### Steps in Model Training
1. Launch the scripting by entering:
```bash
ptyhon train_svm_classifier.py
```
2. The script splits the datase into training and testing sets to evaluate model performance. 
3. Upon completion, an output file called `svm_model.p` will be creating containing the trained SVM model.


## Installation
To run the ASL recognition system, follow these steps to set up your environment and start the application.

1. Clone the repository to your local machine:
```bash
git clone https://github.com/avallip443/asl-recognition.git
```
2. Create a virtual environment:
```bash
python3 -m venv venv
venv\Scripts\activate    # On macOS/Linux use: source venv/bin/activate 
```
3. Install requirements (skip this step if `model-requirements.txt` was executed):
```bash
pip install -r requirements.txt
```
4. Run the application:
```bash
python app.py
```
5. Navigate to the local URL. For example, the URL may be: `http://127.0.0.1:23456`


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
