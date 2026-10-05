import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

# =========================
# PREPARING THE DATA
# =========================

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# =========================
# CONSTANTS
# =========================

IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)

BATCH_SIZE = 128
EPOCHS = 20
CLASSES = 10

VERBOSE = 1
VALIDATION_SPLIT = 0.2

OPTIMIZER = optimizers.RMSprop()

# =========================
# NORMALIZATION
# =========================

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# =========================
# CONVERT LABELS
# =========================

y_train = tf.keras.utils.to_categorical(
    y_train,
    CLASSES
)

y_test = tf.keras.utils.to_categorical(
    y_test,
    CLASSES
)

# =========================
# SIMPLE CNN
# =========================

model = models.Sequential([

    # Input
    layers.Input(shape=INPUT_SHAPE),

    # CONVOLUTION
    layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    # MAX POOLING
    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # DROPOUT
    layers.Dropout(0.25),

    # FLATTEN
    layers.Flatten(),

    # FULLY CONNECTED
    layers.Dense(
        512,
        activation="relu"
    ),

    # DROPOUT
    layers.Dropout(0.5),

    # OUTPUT
    layers.Dense(
        CLASSES,
        activation="softmax"
    )
])

# =========================
# COMPILE
# =========================

model.compile(
    loss="categorical_crossentropy",
    optimizer=OPTIMIZER,
    metrics=["accuracy"]
)

model.summary()


# =========================
# TENSORBOARD
# =========================

callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir="./logs"
    )
]


# =========================
# TRAIN
# =========================

history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks
)


# =========================
# TEST
# =========================

score = model.evaluate(
    X_test,
    y_test,
    verbose=VERBOSE
)

print("\nTest score:", score[0])
print("Test Accuracy:", score[1])