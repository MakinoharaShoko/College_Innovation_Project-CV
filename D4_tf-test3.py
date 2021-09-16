import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


class callBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.2):
            print('\nReached 80%')
            self.model.stop_training = True
cb1 = callBack()

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

plt.imshow(training_images[0])
print(training_images[0])
print((training_labels[0]))

training_images = training_images/255
test_images = test_images/255

training_images = np.expand_dims(training_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)]
)

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy')
model.summary()
model.fit(training_images,training_labels,epochs=15,
          callbacks=[cb1]
          )

model.evaluate(test_images,test_labels)