from keras.preprocessing.image import load_img
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# in this file we are defining and learning our Convolutional Neural Network

# add your path to images folder
base_path = "[...]/images/"
pic_size = 100

for people in os.listdir(base_path + "train/"):
    for i in range(1, 6):
        img = load_img(base_path + "train/" + people + "/" + os.listdir(base_path + "train/" + people)[i],
                       target_size=(pic_size, pic_size))

for expression in os.listdir(base_path + "train"):
    print(str(len(os.listdir(base_path + "train/" + expression))) + " " + expression + " images")

batch_size = 3
data_generator_train = ImageDataGenerator()
data_generator_validation = ImageDataGenerator()

# preparing train and validation data set
train_generator = data_generator_train.flow_from_directory(base_path + "train",
                                                           target_size=(pic_size, pic_size),
                                                           color_mode="grayscale",
                                                           batch_size=batch_size,
                                                           class_mode='categorical',
                                                           shuffle=True)

validation_generator = data_generator_validation.flow_from_directory(base_path + "validation",
                                                                     target_size=(pic_size, pic_size),
                                                                     color_mode="grayscale",
                                                                     batch_size=batch_size,
                                                                     class_mode='categorical',
                                                                     shuffle=False)
# defining model architecture
nb_classes = 3

model = Sequential()
# 1 - Convolution
model.add(Conv2D(512, (3, 3), padding='same', input_shape=(100, 100, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))
# 2 Convolution layer
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
# 3 Convolution layer
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))
# 4 Convolution layer
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))
# 5 Convolution layer
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
# 6 Convolution layer
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))
# 7 Convolution layer
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
# 8 Convolution layer
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))
# 9 Convolution layer
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))
# Flattening
model.add(Flatten())
# Fully connected layer 1st layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))
# Fully connected layer 1st layer
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
# Fully connected layer 1st layer
model.add(Dense(nb_classes, activation='softmax'))

opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# training model
epochs = 20

checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=train_generator.n // train_generator.batch_size,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=validation_generator.n // validation_generator.batch_size,
                              callbacks=callbacks_list
                              )
# you can define you model name beneath
model.save('model_name.model')
