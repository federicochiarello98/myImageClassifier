import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv

from PIL import Image
from keras import layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3

#PRINT THE NUMBER OF IMAGE IN THE DATASET
data_dir = pathlib.Path("DATA/dataset")
training_dir=data_dir/"TRAIN"
val_dir=data_dir/"VAL"
image_count = len(list(data_dir.glob('*/*/*.jpg')))
print(image_count)

#SHOW AN IMAGE EXAMPLE
bulloni = list(training_dir.glob('bulloni/*'))
image=Image.open(str(bulloni[0]))
image.show()

#DEFINE THE IMAGE FEATURES
img_height=640
img_width=640

#DEFINE THE BATCH SIZE
batch_size=4

#DEFINE AUGUMENTATION FEATURES
shear_range = 0.2
zoom_range = 0.2
rotation_range = 50
brightness_range = [0.4,1.5]
horizontal_flip = True

#DEFINE TRAINING PARAMETER
learning_rate = 1e-3
epochs = 30

learning_rate_fine = 3e-4
epochs_fine = 40

#TRAIN SET

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range = shear_range,
                                   zoom_range = zoom_range,
                                   rotation_range= rotation_range,
                                   brightness_range= brightness_range,
                                   horizontal_flip = True)

train_ds = train_datagen.flow_from_directory(training_dir,
                                             shuffle=True,
                                             target_size=(img_height,img_width),
                                             batch_size=batch_size,
                                             class_mode="binary")


#VALIDATION SET
val_datagen = ImageDataGenerator(rescale = 1./255)

val_ds = val_datagen.flow_from_directory(val_dir,
                                         shuffle=True,
                                         target_size=(img_height,img_width),
                                         batch_size=batch_size,
                                         class_mode="binary")

#SHOW AGUMENTATION EXAMPLE
fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(7,7))

for i in range(3):
  image_batch,label_batch = (iter(train_ds)[0])
  first_image = image_batch[0]
  ax[i].imshow(image_batch[0])

plt.show()

#VERIFY NORMALIZATION AND LABELS
image_batch,label_batch = (iter(train_ds)[0])
first_image = image_batch[0]
first_label = label_batch[0]
print(first_label)
plt.imshow(first_image)
print(np.min(first_image), np.max(first_image))

base_model = InceptionV3(weights='imagenet',
                    include_top=False,
                    input_shape=(img_width, img_height, 3))

base_model.trainable=False

inputs = tf.keras.Input(shape=(img_width, img_height, 3))
x = base_model(inputs,training=False)
base_model = tf.keras.Model(inputs,x)

base_model.summary()

model = Sequential([
  base_model,
  #layers.InputLayer(input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dropout(0.2),
  layers.Dense(128, activation='relu'),
  layers.Dense(1,activation='sigmoid')
])


model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

################TRAIN#####################################################

model.summary()

history = model.fit(
  train_ds,
  steps_per_epoch = 169,
  validation_data=val_ds,
  epochs=epochs
)

#################RESULT######################################################

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig("results.png")
plt.show()

#######################FINE TUNING##########################################

base_model.trainable = True

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate_fine),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(
  train_ds,
  steps_per_epoch = 169,
  validation_data=val_ds,
  epochs=epochs_fine
)

acc_fine = history.history['accuracy']
val_acc_fine = history.history['val_accuracy']

loss_fine = history.history['loss']
val_loss_fine = history.history['val_loss']

epochs_range = range(epochs+epochs_fine)

acc = acc+acc_fine
val_acc = val_acc+val_acc_fine

loss = loss+loss_fine
val_loss = val_loss+val_loss_fine

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig("final_results.png")
plt.show()

model.save('Mymodel640.h5')

with open("results.csv",'w') as file:
  writer = csv.writer(file)

  writer.writerow(["EPOCHS","LOSS","ACC","VAL_LOSS","VAL_ACC"])
  for i in epochs_range:
    writer.writerow([str(i),loss[i],acc[i],val_loss[i],val_acc[i]])

