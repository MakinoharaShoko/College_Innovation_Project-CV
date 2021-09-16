import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


class callBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.3):
            print('\nReached 70%')
            self.model.stop_training = True
cb1 = callBack()

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

plt.imshow(training_images[0])
print(training_images[0])
print((training_labels[0]))

training_images = training_images/255
test_images = test_images/255

model = tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(512,activation=tf.nn.relu),
     tf.keras.layers.Dense(10,activation=tf.nn.softmax)]
)

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy')
model.fit(training_images,training_labels,epochs=15,callbacks=[cb1])

model.evaluate(test_images,test_labels)