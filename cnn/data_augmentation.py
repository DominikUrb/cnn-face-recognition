import cv2
import numpy as np
import os
from keras_preprocessing.image import img_to_array, ImageDataGenerator

# this file is making data augmentation - adding some noise to images, rotating them, changing brightness etc,
# which allow you to get much more bigger data set to train your neural network and make it perform better
# in different conditions

# you need to add your directories paths in few places
# add path to folder with your images
for filename in os.listdir('[...]'):
    # blur operation
    img = cv2.imread(filename)
    rows, cols, ch = img.shape
    for i in range(3, 10):
        blur = cv2.blur(img, (i * 1, i * 1))
        # add path to folder where you want to save images
        cv2.imwrite('[...]' + i.__str__() + filename, blur)

    data = img_to_array(img)
    samples = np.expand_dims(data, 0)
    data_generator = ImageDataGenerator(brightness_range=[0.2, 1.7])
    data_generator_z = ImageDataGenerator(zoom_range=[0.6, 1.1])
    it = data_generator.flow(samples, batch_size=1)
    itZ = data_generator_z.flow(samples, batch_size=1)
    data_generator_r = ImageDataGenerator(rotation_range=75)
    itR = data_generator_r.flow(samples, batch_size=1)

    # changing images brightness
    for i in range(8):
        batch = it.next()
        image = batch[0].astype('uint8')
        # add path to folder where you want to save images
        cv2.imwrite('[...]' + i.__str__() + filename, image)

    # zooming images
    for i in range(10):
        batch_z = itZ.next()
        image_z = batch_z[0].astype('uint8')
        # add path to folder where you want to save images
        cv2.imwrite('[...]' + i.__str__() + filename, image_z)

    # zooming images
    for i in range(10):
        batchR = itR.next()
        image_r = batchR[0].astype('uint8')
        # add path to folder where you want to save images
        cv2.imwrite('[...]' + i.__str__() + filename, image_r)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
