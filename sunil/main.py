import os
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import Buffer
from datetime import datetime
import csv
import time
from pyautogui import press, typewrite, hotkey, doubleClick
import threading
ret1 = None
predictions = []
LABELS = []
DUR = 3

EXPECTED_LABELS = ["thumb up", "thumb down", "swiping left", "swiping right", "sliding two fingers down", "sliding two fingers up", "zooming in with two fingers", "zooming out with two fingers"]

with open("jester-v1-labels.csv", "r") as csvfile:
    csv_data = csv.reader(csvfile, delimiter=',')
    for row in csv_data:
        LABELS.append(row[0]);

class PredictionThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
     

    def perform_action(self, label):
       
        if label not in EXPECTED_LABELS:
            return
        
        print("########### PREDICTED LABEL : {}".format(label))
        global ret1
        
        if "thumb" in label:
            print("play/pause")
            press('space')
            ret1 = 1
        elif "down" in label:
            press('pagedown') 
            print('down')
            ret1 = 1 
        elif "up" in label:
            press('pageup')
            print('up')
            ret1 = 1
        elif "left" in label:
            press("left")
            print("left")
            ret1 = 1
        elif "right" in label:
            press("right")
            print("right")
            ret1 = 1
        elif " in " in label:
            # hotkey('ctrl', 'add')
            doubleClick();
            # find key comb for zoom in
            print("zoom in")
        elif " out " in label:
            # hotkey('ctrl', 'subtract')
            doubleClick(button='right')
            print("zoom out")


    def run(self):
        global predictions
        predictions1 = getClassProbabilities(predictions);
                # print(predictions1)
        predicted_label = predictions1[0][0].lower();

        if predicted_label in RESLABELS.keys():
            RESLABELS[predicted_label] += 1
        else:
            RESLABELS[predicted_label] = 1
        prev = check[0]
        samecnt = check[1]
        if CNT == DUR:
            arr = [e[0] for e in sorted(list(RESLABELS.items()), key=lambda e: e[1],reverse=True)[:5]]
            # print("{}".format(arr))
            label = arr[0]
            if label == prev or (check[1] == 2 and prev in arr):
                check[1] += 1
            else:
                check[1] = 1
                check[0] = label
            if check[1] == 3:
                self.perform_action(prev)
                # print("###PREDICTED : {}".format(prev))
                check[1] = 0
                check[0] = ''
            
            # print(sorted(list(RESLABELS.items()), key=lambda e: e[1],reverse=True)[0][0])
        # if perform_action(class_probabilities[0][0].lower()):
        #     pass
            # time.sleep(3) # to avoid false repeat 
            

def getClassProbabilities(arr):
    res = []
    for i in range(len(arr)):
        res.append((LABELS[i], arr[i]));
    res.sort(key = lambda e: e[1], reverse=True);
    return res[:5]

             
def modelTest(new_model, buf, frame, check, CNT, RESLABELS):
    buf.addFrame(frame)
    # print(buf.size)
    global predictions

    if buf.size == 30:
        intensor = buf.getTensor();
        intensor = tf.expand_dims(intensor, 0);
        predictions = new_model.predict(intensor, steps=1)[0]
        thread = PredictionThread()
        thread.start()

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
    RESLABELS = {}
    CNT = 0
    check = [0, '']
    # print("initializing")
    # time.sleep(3)

    while True:
        if ret1 == 1:
            buf.clear();
            time.sleep(3)
            ret1 = None
        ret, frame = vid.read()
        x = threading.Thread(target=modelTest, args=(new_model, buf, frame, check, CNT, RESLABELS))
        if buf.size == 30:
                CNT += 1
            # buf.clear();
        if CNT > DUR:
            CNT = 0
            RESLABELS = {}
           
        x.start()
        frame2 = cv2.flip(frame, 1)
        cv2.imshow('frame', frame2)
        
        x.join()
        # q is the quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # time.sleep(0.01)

    vid.release()
    cv2.destroyAllWindows()
