from recognizer import Recognizer
from craft import NpPointsCraft
from utils import crop_number_plate_zones_from_images, unzip


import os
import tensorrt as trt

import cv2

def start_model(craft_engine_path, recognizer_engine_path):
    craft = NpPointsCraft(craft_engine_path)
    recognizer = Recognizer(recognizer_engine_path)

    return craft, recognizer

def inference(model1, model2, input):
    img = input['general_frame']
    images_points, _ = model1.detect(unzip(
            [[img], input['bbox']]
    ))

    results, _ = crop_number_plate_zones_from_images([img], images_points)

    prediction = None
    for result in results:
        prediction = model2.predict(result)
    
    if prediction is None or len(prediction) == 0:
        return "00000000"
    return prediction['label']
