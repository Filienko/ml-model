import tensorflow as tf
from tensorflow import keras
from keras import layers

# Settings
img_height = 32 # Set Image Height
img_width = 32 # Set Image Width
batch_size = 32
seed = 9867787
epochs = 15

# Path to training data
path = "image_postchange/"

# Dataset Creation
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	path,
	validation_split=0.2,
	subset="training",
	seed=seed,
	image_size=(img_height, img_width),
	batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
	validation_split=0.2,
	subset="validation",
	seed=seed,
	image_size=(img_height, img_width),
	batch_size=batch_size
)

class_names = train_ds.class_names

# Configure for Performace
# https://www.tensorflow.org/tutorials/images/classification
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

# Randomization to Prevent Overfitting
data_augmentation = keras.Sequential([
	layers.RandomFlip("horizontal",input_shape=(img_height,img_width,3)),
	layers.RandomRotation(0.1),
	layers.RandomZoom(0.1),
])



model = keras.Sequential([
	data_augmentation,
	layers.Rescaling(1./255),
	layers.Conv2D(16, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(32, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(64, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Dropout(0.2),
	layers.Flatten(),
	layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name="outputs")
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.summary()

history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

model.save('image_model')