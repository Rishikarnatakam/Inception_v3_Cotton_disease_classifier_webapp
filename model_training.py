import tensorflow as tf
import os
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from keras.optimizers import AdamW
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout

# Set image size
IMAGE_SIZE = [224, 224]
batch_size = 32

# Define train and test paths - updated to local paths
train_path = 'Dataset/Train'
test_path = 'Dataset/Test'

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range=45,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   brightness_range=[0.5, 1.5],
                                   vertical_flip=True,
                                   horizontal_flip = True,
                                   )
test_datagen = ImageDataGenerator(rescale = 1./255,)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = IMAGE_SIZE,
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = IMAGE_SIZE,
                                            batch_size = batch_size,
                                            class_mode = 'categorical')

# Load InceptionV3 with pre-trained weights
inception = InceptionV3(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

# Freeze the first few layers
for layer in inception.layers[:200]:
    layer.trainable = False

# Add additional layers
x = GlobalAveragePooling2D()(inception.output)
x = Flatten()(x)
x = Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.4)(x)  # Increase dropout rate
x = Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.4)(x)  # Increase dropout rate
x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.4)(x)  # Increase dropout rate
num_classes = len(training_set.class_indices)
prediction = Dense(num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=inception.input, outputs=prediction)

# Compile the model with reduced learning rate for fine-tuning
optimizer = AdamW(learning_rate=0.0001, weight_decay=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Apply early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Define the initial number of epochs
initial_epochs = 20

# Train the model for the initial number of epochs
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=initial_epochs,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set),
    callbacks=[early_stopping]
)

# Plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# Plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

# Save the model
model.save("finetune3.h5")