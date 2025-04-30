import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# load data
data_dict = pickle.load(open('./new_data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# normalize data
data = data / 255.0

# create training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# label encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)

# data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

num_classes = len(np.unique(labels))

def create_model(input_shape=(224, 224, 3), num_classes=len(np.unique(labels))):
# def create_model():
    model = models.Sequential([
        Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax') 
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# create model
model = create_model(input_shape=(224, 224, 3))

# display model architecture
model.summary()

# 5-fold stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1

for train_index, val_index in skf.split(x_train, y_train):
    print(f"Training fold {fold_no}...")
    
    # split data for this fold
    x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Train with data augmentation
    model.fit(
        datagen.flow(x_train_fold, y_train_fold, batch_size=64),
        steps_per_epoch=len(x_train_fold) // 64,
        epochs=10,
        validation_data=(x_val_fold, y_val_fold),
        verbose=1
    )
    
    fold_no += 1

# create predictions
y_predict = model.predict(x_test)

# Ensure that predictions are classified properly
y_predict = np.argmax(y_predict, axis=1)

print(f"Accuracy: {accuracy_score(y_test, y_predict) * 100:.2f}%")
print(classification_report(y_test, y_predict))

model.save('new_model.h5')
