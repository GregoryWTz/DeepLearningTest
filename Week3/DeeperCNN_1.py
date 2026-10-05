import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, optimizers

# =========================
# PREPARING THE DATA
# =========================

EPOCHS = 50
NUM_CLASSES = 10

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

mean = np.mean(X_train, axis=(0, 1, 2, 3))
std = np.std(X_train, axis=(0, 1, 2, 3))

X_train = (X_train - mean) / (std + 1e-7)
X_test = (X_test - mean) / (std + 1e-7)

y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

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

model = build_model()

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(),
    metrics=["accuracy"]
)

model.summary()

BATCH_SIZE = 64

history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test)
)

score = model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE
)

print("\nTest score:", score[0])
print("Test Accuracy:", score[1])