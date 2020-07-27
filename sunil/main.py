import os
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import Buffer
from datetime import datetime
import csv
import time

LABELS = []

with open("jester-v1-labels.csv", "r") as csvfile:
    csv_data = csv.reader(csvfile, delimiter=',')
    for row in csv_data:
        LABELS.append(row[0]);

def getClassProbabilities(arr):
    res = []
    for i in range(len(arr)):
        res.append((LABELS[i], arr[i]));
    res.sort(key = lambda e: e[1], reverse=True);
    return res[:5]

if __name__ == "__main__":
    # print(tf.version.VERSION)

    # Recreate the exact same model, including its weights and the optimizer
    new_model = tf.keras.models.load_model('test.h5')
    # create a video object capturing from device with id 0
    vid = cv2.VideoCapture(0)
    buf = Buffer.Buffer()

    # Show the model architecture
    # new_model.summary()

    while True:
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        buf.addFrame(frame)
        # print(buf.size)

        if buf.size == 30:
             intensor = buf.getTensor();
             intensor = tf.expand_dims(intensor, 0);
             intensor = tf.expand_dims(intensor, -1);
             predictions = new_model.predict(intensor, steps=1)[0]
             class_probabilities = getClassProbabilities(predictions)
             if "other things" not in class_probabilities[0][0].lower():
                print(class_probabilities[0]);
        

        # q is the quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    vid.release()
    cv2.destroyAllWindows()
