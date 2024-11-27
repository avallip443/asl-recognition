import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping


# load data
data_dict = pickle.load(open('./new_data.pickle', 'rb'))
data = np.asarray(data_dict['data'])  # shape: (samples, 21, 2)
labels = np.asarray(data_dict['labels'])  # shape: (samples,)

# normalize data to [0, 1] range
data = data / 255.0

# create training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Label encoding for class labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Print shapes for debugging
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

# Define MLP model architecture
def create_mlp(input_shape=(21, 2), num_classes=len(np.unique(labels))):
    model = models.Sequential([
        Input(shape=input_shape),  # Input shape: (21, 2)
        layers.Flatten(),  # Flatten into 1D vector
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),  # Dropout for regularization
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')  # Output layer
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create MLP model
model = create_mlp(input_shape=(21, 2))

# Display model architecture
model.summary()

# EarlyStopping callback to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 5-fold stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1

for train_index, val_index in skf.split(x_train, y_train):
    print(f"Training fold {fold_no}...")
    
    # Split data for this fold
    x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Train the MLP model
    model.fit(
        x_train_fold, y_train_fold,
        batch_size=64,
        epochs=10,
        validation_data=(x_val_fold, y_val_fold),
        callbacks=[early_stopping],
        verbose=1
    )

    fold_no += 1

# Create predictions on the test set
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)  # Convert probabilities to class indices

# Evaluate model performance
print(f"Accuracy: {accuracy_score(y_test, y_predict) * 100:.2f}%")
print(classification_report(y_test, y_predict))

# Save the trained model
model.save('new_model.h5')
