# ASL Recognition System

## Description
This project implements an ASL recognition system using computer vision and machine learning. It processes hand landmarks detected by Mediapipe and trains a model to recognize gestures.

## Table of Contents
- [Data Collection](#data-collection)
- [Dataset Creation](#dataset-creation)
- [Dependencies](#dependencies)
- [Installation](#installation)


## Data Collection
The script `collect_data.py` collect images from a webcam to create a dataset for ASL gestures (classes). Here's how it works:

### Set Up 
Before running the script, it is recommended to use a virtual environment.

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
5. Run the script OR
```bash
python collect_data.py
```

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

### Image Collection Process
Once the script is running, it will prompt you to press keys to start collecting image data for each gesture (class). The data is captured via your webcamera.
- Each image set will be stored in the appropriate subdirectory in the `./data` directory (e.g. `./data/0` for class 0).
- Images collected are specified by the total number of classes (`NUMBER_OF_CLASSES`) and the number of images per class (`DATASET_SIZE`).
- The webcam feed will display instructions. Press "1" to start collecting images for the current class or "q" to quit.


## Dataset Creation
The script `create_dataset.py` processes the image sets to create a dataset of hand landmarks for training ASL gesture recognition models. It uses `Mediapipe` to detect hand landmarks from image and saves the extracted data and its corresponding labels to a `.pickle` file.

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
