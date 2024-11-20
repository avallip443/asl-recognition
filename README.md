# ASL Recognition System

## Description
This project implements an ASL recognition system using computer vision and machine learning. It processes hand landmarks detected by Mediapipe and trains a model to recognize gestures.

## Data Collection
The script `collect_data.py` collected image data from a webcam to create a dataset for a specified number of gestures (classes). Here's how it works:

### Overview
Each dataset will be saved in the `./data` directory with each gesture (class) having its own subdirectory (e.g. `./data/0`).
A user can specify:
- The total number of classes (`NUMBER_OF_CLASSES`).
- The number of images per class (`DATASET_SIZE`).

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

### Data Collection Process


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
