import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import to_categorical
import numpy as np


# Define a CNN model architecture
def create_model_landmarks():
    model = models.Sequential([
        Input(shape=(21, 2)),  # Input shape for hand landmarks
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(label_encoder.classes_), activation='softmax')  # Number of classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Load data
data_dict = pickle.load(open('./kaggle_data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Normalize data
data = data / 255.0
print('Data shape:', data.shape)

# Create training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Prepare data
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_train = to_categorical(y_train, num_classes=len(label_encoder.classes_))
y_test = to_categorical(y_test, num_classes=len(label_encoder.classes_))

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")

# 5-fold stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
all_accuracies = []

for train_index, val_index in skf.split(x_train, label_encoder.inverse_transform(np.argmax(y_train, axis=1))):
    print(f"Training fold {fold_no}...")
    
    # Split data for this fold
    x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Create a new model instance
    model = create_model_landmarks()
    
    # Display model architecture
    print(f"\nModel architecture for fold {fold_no}:")
    model.summary()
    
    # Train the model
    history = model.fit(
        x_train_fold,
        y_train_fold,
        batch_size=64,
        epochs=10,
        validation_data=(x_val_fold, y_val_fold),
        verbose=1
    )
    
    # Evaluate fold accuracy
    val_accuracy = model.evaluate(x_val_fold, y_val_fold, verbose=0)[1]
    print(f"Validation Accuracy for fold {fold_no}: {val_accuracy * 100:.2f}%")
    all_accuracies.append(val_accuracy)
    
    fold_no += 1

# Cross-validation accuracy summary
print(f"Cross-validation accuracies: {[acc * 100 for acc in all_accuracies]}")
print(f"Mean Accuracy: {np.mean(all_accuracies) * 100:.2f}%")

# Create predictions
y_predict = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print(f"Test Accuracy: {accuracy_score(y_true, y_predict) * 100:.2f}%")
print(classification_report(y_true, y_predict))

# Save model
model.save('kaggle_model.h5')
