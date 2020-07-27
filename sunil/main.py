import os
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import Buffer
from datetime import datetime
import csv
import time
from pyautogui import press, typewrite, hotkey
import threading



LABELS = []

with open("jester-v1-labels.csv", "r") as csvfile:
    csv_data = csv.reader(csvfile, delimiter=',')
    for row in csv_data:
        LABELS.append(row[0]);


def perform_action(label):
    if "thumb" in label:
        print("space")
        press('space')
    elif "down" in label:
        press('down') 
        print('down') 
    elif "up" in label:
        press('up')
        print('up')
    elif "left" in label:
        press("left")
        print("left")
    elif "right" in label:
        press("right")
        print("right")

        

def getClassProbabilities(arr):
    res = []
    for i in range(len(arr)):
        res.append((LABELS[i], arr[i]));
    res.sort(key = lambda e: e[1], reverse=True);
    return res[:5]


def modelTest(new_model, buf, frame):
    buf.addFrame(frame)
        # print(buf.size)

    if buf.size == 30:
            intensor = buf.getTensor();
            intensor = tf.expand_dims(intensor, 0);
            intensor = tf.expand_dims(intensor, -1);
            predictions = new_model.predict(intensor, steps=1)[0]
            class_probabilities = getClassProbabilities(predictions)
            perform_action(class_probabilities[0][0].lower())
    

if __name__ == "__main__":
    # print(tf.version.VERSION)

    # Recreate the exact same model, including its weights and the optimizer
   
    # create a video object capturing from device with id 0
    vid = cv2.VideoCapture(0)
    print("width: {} height: {}".format(vid.get(3), vid.get(4)));
    buf = Buffer.Buffer()
    new_model = tf.keras.models.load_model('test.h5')

    # Show the model architecture
    # new_model.summary()

    while True:
        ret, frame = vid.read()
        x = threading.Thread(target=modelTest, args=(new_model, buf, frame))
        x.start()
        frame2 = cv2.flip(frame, 1)
        cv2.imshow('frame', frame2)
        
        x.join()
        # q is the quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    vid.release()
    cv2.destroyAllWindows()
