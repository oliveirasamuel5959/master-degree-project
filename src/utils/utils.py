import cv2
import numpy as np

class ComputerVision:
    def __init__(self):
        pass
      
    def calculate_brightness(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = hsv[:, :, 2].mean()  # Mean brightness of the V channel
        return brightness

    def classify_frame(self, brightness, thresholds):
        if brightness > thresholds['day']:
            return 'Day'
        elif thresholds['evening'] < brightness <= thresholds['day']:
            return 'Evening'
        else:
            return 'Night'