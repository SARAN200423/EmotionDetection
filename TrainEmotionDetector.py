import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
import logging
import sys

# Ensure UTF-8 encoding for terminal output
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Set environment variable to use UTF-8 encoding for I/O
os.environ["PYTHONIOENCODING"] = "utf-8"

# Logging setup (logs key milestones only)
log_file = 'training_log.txt'
logging.basicConfig(filename=log_file, level=logging.INFO, encoding='utf-8')

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# Custom data loader with tf.data.Dataset
def load_data(data_dir, batch_size=64, img_size=(48, 48)):
    """
    Function to load data using tf.data.Dataset and image preprocessing.
    """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="grayscale",
        label_mode="categorical"
    )
    return dataset

# Load the train and validation datasets using tf.data API
train_dataset = load_data('D:/cv/archive (4)/train')
validation_dataset = load_data('D:/cv/archive (4)/test')

# Define the model architecture
emotion_model = Sequential([
    Input(shape=(48, 48, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile the model
emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Learning rate reduction callback
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=1e-6)

# Log the start of training
logging.info("Training started.")

# Train the model and show epoch progress in the terminal
emotion_model.fit(
    train_dataset,
    epochs=80,
    validation_data=validation_dataset,
    callbacks=[lr_reduction],
    verbose=1  # Shows epoch progress in the terminal
)

# Log the completion of training
logging.info("Training finished.")

# Save the model weights and architecture
emotion_model.save_weights('emotion_model_weights.h5')
logging.info("Model weights saved to 'emotion_model.weights.h5'.")

# Save the full model, including architecture, weights, and optimizer state
emotion_model.save('emotion_model.h5')
logging.info("Full model saved to 'emotion_model.h5'.")

# Save the model architecture to a JSON file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w", encoding='utf-8') as json_file:
    json_file.write(model_json)
logging.info("Model architecture saved to 'emotion_model.json'.")
