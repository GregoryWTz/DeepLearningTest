import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


# =========================
# CONSTANTS
# =========================

EPOCHS = 50
NUM_CLASSES = 10
BATCH_SIZE = 64


# =========================
# LOAD DATA
# =========================

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# =========================
# PREPARE DATA
# =========================

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")


# Normalize
mean = np.mean(X_train, axis=(0, 1, 2, 3))
std = np.std(X_train, axis=(0, 1, 2, 3))

X_train = (X_train - mean) / (std + 1e-7)
X_test = (X_test - mean) / (std + 1e-7)


# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(
    y_train,
    NUM_CLASSES
)

y_test = tf.keras.utils.to_categorical(
    y_test,
    NUM_CLASSES
)


# =========================
# BUILD MODEL
# =========================

def build_model():

    model = models.Sequential()

    # =========================
    # 1ST BLOCK
    # =========================

    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            input_shape=X_train.shape[1:],
            activation="relu"
        )
    )

    model.add(layers.BatchNormalization())

    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        )
    )

    model.add(layers.BatchNormalization())

    model.add(
        layers.MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    model.add(layers.Dropout(0.2))


    # =========================
    # 2ND BLOCK
    # =========================

    model.add(
        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        )
    )

    model.add(layers.BatchNormalization())

    model.add(
        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        )
    )

    model.add(layers.BatchNormalization())

    model.add(
        layers.MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    model.add(layers.Dropout(0.3))


    # =========================
    # 3RD BLOCK
    # =========================

    model.add(
        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu"
        )
    )

    model.add(layers.BatchNormalization())

    model.add(
        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        )
    )

    model.add(layers.BatchNormalization())

    model.add(
        layers.MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    model.add(layers.Dropout(0.4))


    # =========================
    # OUTPUT
    # =========================

    model.add(layers.Flatten())

    model.add(
        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    )

    return model


# =========================
# CREATE MODEL
# =========================

model = build_model()


# =========================
# COMPILE
# =========================

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(),
    metrics=["accuracy"]
)

model.summary()


# =========================
# DATA AUGMENTATION
# =========================

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)


# =========================
# TRAIN
# =========================

history = model.fit(
    datagen.flow(
        X_train,
        y_train,
        batch_size=BATCH_SIZE
    ),
    epochs=EPOCHS,
    verbose=1,
    validation_data=(X_test, y_test)
)


# =========================
# TEST
# =========================

score = model.evaluate(
    X_test,
    y_test,
    batch_size=128,
    verbose=1
)

print(
    "\nTest result: %.3f%% accuracy, %.3f loss"
    % (score[1] * 100, score[0])
)