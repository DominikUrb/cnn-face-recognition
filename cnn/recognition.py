import cv2
import numpy as np
import tensorflow as tf
import os

# this file allow you to load your CNN model and recognize faces

# loading caffe model for face detection
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
cap = cv2.VideoCapture(0)

# beneath add names of persons which you have learned model to recognize
CATEGORIES = ["person1  ", "person2  ", "person3  "]
# add some path (could me anywhere) samples of faces are going to be saved there, recognized or not and then overwritten
path_output = '[...]'


# preparing samples to enter the neural network
def prepare(filepath):
    img_size = 100
    new_array = cv2.resize(gray_image, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 1)


# loading model - add name of your model
model = tf.keras.models.load_model("[...].model")

while True:
    # face detection
    _, frame = cap.read()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # preparing image to prediction
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        cropped = frame[startY - h:endY, startX - w:endX]
        cropped_photo = cv2.imwrite(os.path.join(path_output, 'elo.jpg'), cropped)
        gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # prediction
        prediction = model.predict(prepare(cropped_photo))
        accuracy = np.amax(prediction)

        if accuracy < 0.8:
            text = 'Unknown'
        elif accuracy > 0.8:
            text = CATEGORIES[np.argmax(prediction)] + round(accuracy * 100, 2).__str__()

        # printing prediction on screen
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.imshow('Face recognition', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
