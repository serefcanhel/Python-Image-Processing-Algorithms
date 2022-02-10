import cv2
import numpy as np
def on_mouse(event, x, y,image,clicks, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Seed: ' + str(x) + ', ' + str(y), image[y, x])
        clicks.append((y, x))