import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

(x_train, _), (_, _) = mnist.load_data()

x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=3)

generator = Sequential()
generator.add(Dense(7 * 7 * 128, input_dim=100))
generator.add(LeakyReLU())
generator.add(Reshape((7, 7, 128)))
generator.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
generator.add(LeakyReLU())
generator.add(Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='tanh'))

discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=(28, 28, 1)))
discriminator.add(LeakyReLU())
discriminator.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
discriminator.add(LeakyReLU())
discriminator.add(Flatten())
discriminator.add(Dropout(0.4))
discriminator.add(Dense(1, activation='sigmoid'))

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

epochs = 50
batch_size = 128
steps_per_epoch = x_train.shape[0] // batch_size

for epoch in range(epochs):
    for step in range(steps_per_epoch):
        noise = np.random.normal(0, 1, size=(batch_size, 100))
        real_images = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

        generated_images = generator.predict(noise)

        X = np.concatenate([real_images, generated_images])
        y_dis = np.zeros(2 * batch_size)
        y_dis[:batch_size] = 0.9

        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(X, y_dis)

        noise = np.random.normal(0, 1, size=(batch_size, 100))
        y_gen = np.ones(batch_size)

        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, y_gen)

    print(f'Epoch: {epoch+1}/{epochs}, Ayırt Edici Kayıp: {d_loss[0]}, Ayırt Edici Doğruluk: {d_loss[1]*100:.2f}%, Üreteci Kayıp: {g_loss}')
