import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np

# load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# normalize data
data = data / np.max(data)

# flatten the data 
data = data.reshape(data.shape[0], -1)

# create training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train model 
def create_model():
    model = models.Sequential([
        Input(shape=(128, 128, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(labels), activation='sigmoid') # Sigmoid for number of classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# create instance of new model
model = create_model()

# train classifers
model.fit(x_train, y_train)

# create predictions
y_predict = model.predict(x_test)

print(f"Accuracy: {accuracy_score(y_test, y_predict) * 100:.2f}%")
print(classification_report(y_test, y_predict))

# save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)