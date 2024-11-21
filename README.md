# ASL Recognition System

## Description
This project implements an ASL recognition system using computer vision and machine learning. It processes hand landmarks detected by Mediapipe and trains a model to recognize gestures.

## Table of Contents
- [Data Collection](#data-collection)
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
5. Run the script
```bash
python collect_data.py
```

### Image Collection Process
Once the script is running, it will prompt you to press keys to start collecting image data for each gesture (class). The data is captured via your webcamera.
- Each image set will be stored in the appropriate subdirectory in the `./data` directory (e.g. `./data/0` for class 0).
- Images collected are specified by the total number of classes (`NUMBER_OF_CLASSES`) and the number of images per class (`DATASET_SIZE`).
- The webcam feed will display instructions. Press "1" to start collecting images for the current class or "q" to quit.


## Dependencies
The following libraries are required to run the project:
- `os`: Built-in Python module for file and directory management (no installation required).
- `pickle`: Built-in Python module for serializing and de-serializing objects (no installation required).
- `mediapipe`: For hand landmarks detection.
- `matplotlib`: For plotting and visualization.
- `numpy`: For numerical computations.
- `opencv-python`: For image and video processing.
- `scikit-learn`: For machine learning algorithms.

## Installation

Install the required libraries with the following command:

```bash
pip install opencv-python mediapipe matplotlib numpy scikit-learn
```
