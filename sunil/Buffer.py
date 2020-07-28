import collections 
import os
import tensorflow as tf
from tensorflow import keras
import cv2

class Buffer:
    def __init__(self):
        self._stack = None
        self.size = 0

    def transform(self, img):
        #transform image to 64x64 2d tensor
        frame = cv2.resize(img, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
        rgb_tensor = tf.expand_dims(tf.convert_to_tensor(frame, dtype=tf.float32), -1)
        rgb_tensor = tf.image.per_image_standardization(rgb_tensor)
        return rgb_tensor


    def pushBack(self, img):
        tensor = self.transform(img);
        tensor = tf.expand_dims(tensor, 0);

        if self._stack == None:
            self._stack = tensor; 
        else:
            self._stack = tf.concat([self._stack, tensor], 0)

        self.size += 1

    def popFront(self):
        self._stack = tf.slice(self._stack, [1, 0, 0, 0], [30, 64, 64, 1]);
        print(self._stack.shape);
        self.size -= 1

    def addFrame(self, img):
        self.pushBack(img)
        if self.size >= 31:
            self.popFront()

    def getTensor(self):
        #create 3d tensor from the buffer
        return self._stack

        

