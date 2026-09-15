import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

# =========================
# PREPARING THE DATA
# =========================

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

# Reshape
X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert to float32
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

# Number of classes
NB_CLASSES = 10

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)


# =========================
# CONSTRAINTS
# =========================

EPOCHS = 5
BATCH_SIZE = 128

IMG_ROWS = 28
IMG_COLS = 28

INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)

OPTIMIZER = optimizers.Adam()

VALIDATION_SPLIT = 0.2

VERBOSE = 1


# =========================
# LENET
# =========================

model = models.Sequential([
    
    layers.Input(shape=INPUT_SHAPE),

    layers.Conv2D(6, (5, 5), activation="relu"),

    layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ),

    layers.Conv2D(16, (5, 5), activation="relu"),

    layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ),

    layers.Flatten(),

    layers.Dense(120, activation="relu"),

    layers.Dense(84, activation="relu"),

    layers.Dense(NB_CLASSES, activation="softmax")
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