import cv2
import numpy as np
import os

# this file is detecting faces in your images data and excising them

# face detecting network
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
for filename in os.listdir('/Users/kubab/PycharmProjects/ProjektInd/ZDJ/'):
    filename2 = ('/Users/kubab/PycharmProjects/ProjektInd/it/' + filename)
    img = cv2.imread(filename2)
    image = cv2.resize(img, (300, 300))
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    # face detecting
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # face excision
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cropped = image[startY - w:endY, startX - h:endX]

    # saving file with excised face
    cv2.imwrite(filename, cropped)
    cv2.waitKey(0)
