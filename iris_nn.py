# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout

# Import Data
df_iris = pd.read_csv("data/Iris.csv")

# Get Features
df_features = df_iris.drop(columns = ['Species', 'Id'])
df_labels = df_iris['Species']

x = np.array(df_features)
y = np.array(df_labels)

# Label Encoding
df_y = pd.get_dummies(y)
y_encoded = np.array(df_y)

# Data Splitting
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size = 0.2, random_state = 42)

# Model Building
model = Sequential()
model.add(Input(shape = x_train[0].shape))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

# Model Training
history = model.fit(x_train,
          y_train,
          validation_data = (x_test, y_test),
          epochs = 300)

# Make Prediction
labels = set(df_iris['Species'].tolist())
pred = [np.argsort(i) for i in model.predict(x)]
pred = [i[2] for i in pred]
pred = [list(labels)[i] for i in pred]

df_iris['Prediction'] = pred
df_iris.to_csv('Iris_predicted.csv', index = False)