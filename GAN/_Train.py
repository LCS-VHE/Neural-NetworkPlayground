import os
import tensorflow as tf
from tensorflow import keras as keras


def transformation(image, _):
    img = image / 255
    return (img, img)


def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


train_data = os.path.join(os.curdir, "Data", "Train")
train_images = tf.keras.preprocessing.image_dataset_from_directory(train_data, image_size=(64, 64), shuffle=True,
                                                                   color_mode='rgb', batch_size=128)

test_data = os.path.join(os.curdir, "Data", "Valid")
valid_images = tf.keras.preprocessing.image_dataset_from_directory(test_data, image_size=(64, 64), shuffle=True,
                                                                   color_mode='rgb', batch_size=128)
train_images = train_images.map(transformation)
valid_images =  valid_images.map(transformation)

conv_encoder = keras.models.Sequential([
    keras.layers.Reshape([64, 64, 3], input_shape=[64, 64, 3]),
    keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='selu'),

    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='selu'),

    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='selu'),
    keras.layers.MaxPool2D(pool_size=2),

    keras.layers.Conv2D(64 * 2, kernel_size=3, padding='same', activation='selu'),
    keras.layers.MaxPool2D(pool_size=2),  # Output shape will be (4, 4, 128)
])

conv_decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='selu',
                                 input_shape=[4, 4, 128]),
    keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='selu'),
    keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding='same', activation='selu'),
    keras.layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', activation='selu'),
    keras.layers.Reshape([64, 64, 3])
])

stacked_autoencoder = keras.models.Sequential([conv_encoder, conv_decoder])
opt = keras.optimizers.Adam(learning_rate=0.01)
stacked_autoencoder.compile(loss='mse', metrics=[rounded_accuracy], optimizer=opt)

best_model = keras.callbacks.ModelCheckpoint(f"Best Model.h5", save_best_only=True)
epochs = 1000

best_model = keras.callbacks.ModelCheckpoint(f"Best Model.h5", save_best_only = True)
epochs = 1000

stacked_autoencoder.fit(train_images, epochs=epochs, callbacks=best_model, validation_data=valid_images)