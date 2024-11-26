import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np

# load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# normalize data
data = data / 255.0

# flatten the data 
data = data.reshape(data.shape[0], -1)

# create training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)

# define a CNN model architecture 
def create_model():
    model = models.Sequential([
        Input(shape=(512, 512, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(labels), activation='sigmoid') # Sigmoid for number of classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 5-fold stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1

for train_index, val_index in skf.split(x_train, y_train):
    print(f"Training fold {fold_no}...")
    
    # Split data for this fold
    x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Create a new model instance
    model = create_model()
    
    # Display model architecture
    print(f"\nModel architecture for fold {fold_no}:")
    model.summary()
    
    # train classifers
    model.fit(
        x_train_fold,
        y_train_fold,
        batch_size=64,
        epochs=10,
        validation_data=(x_val_fold, y_val_fold),
        verbose=1
    )
    
    fold_no += 1

# create predictions
y_predict = model.predict(x_test)

print(f"Accuracy: {accuracy_score(y_test, y_predict) * 100:.2f}%")
print(classification_report(y_test, y_predict))

# save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)