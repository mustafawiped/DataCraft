import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

data = pd.read_csv('veri_seti.csv')
features = data.drop(['etiket'], axis=1).values
labels = data['etiket'].values

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

categorical_labels = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(scaled_features, categorical_labels, test_size=0.2, random_state=42)

model = Sequential()
model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(len(np.unique(labels)), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

_, test_accuracy = model.evaluate(X_test, y_test)
print('Doğruluk Oranı:', test_accuracy)
