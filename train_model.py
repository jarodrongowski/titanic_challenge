import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers, models, Input, callbacks, optimizers, Model

plt.rcParams['font.size'] = 10.0
plt.rcParams['font.weight'] = 'medium'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.figsize'] = (0.5*19.20, 0.5*10.80)

def read_data(file_path):
    data = pd.read_csv(file_path)
    data_reduced = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    data_reduced['Sex'] = data_reduced['Sex'].map({'male': 0, 'female': 1})
    data_reduced['Age_missing'] = data_reduced['Age'].isna().astype(int)
    data_reduced['Age'] = data_reduced['Age'].fillna(data_reduced['Age'].median())
    data_reduced = pd.get_dummies(data_reduced, columns=['Embarked'])
    data_reduced['Embarked_C'] = data_reduced['Embarked_C'].map({True: 1, False: 0})
    data_reduced['Embarked_Q'] = data_reduced['Embarked_Q'].map({True: 1, False: 0})
    data_reduced['Embarked_S'] = data_reduced['Embarked_S'].map({True: 1, False: 0})

    scaler = StandardScaler()
    data_reduced[['Age', 'Fare']] = scaler.fit_transform(data_reduced[['Age', 'Fare']])
    
    #print(data_reduced.head(n=40))
    
    return data_reduced

df = read_data('train.csv')

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train.values
X_val = X_val.values
y_train = y_train.values
y_val = y_val.values

model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16
)

loss, acc = model.evaluate(X_val, y_val)
model.save('titanic_challenge_model.keras')

history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure()
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.savefig('acc.png', dpi=400)

plt.figure()
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.savefig('loss.png', dpi=400)