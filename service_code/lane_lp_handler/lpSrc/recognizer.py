import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
from typing import Dict
import cv2


INDEX_TO_COUNTRY: Dict[int, str] = {
    0: "kz",
    1: "ru",
    2: "kg",
    3: "ua",
    4: "by",
    5: "eu",
    6: "ge",
    7: "am",
    8: "unknown",
    9: "military",
    10: "uz",
    11: "mn",
    12: "md",
    13: "cn",
    14: "ir",
    15: "tr"
}


class Recognizer():
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f:
            recognizer_engine_data = f.read()

        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        recognizer_engine = runtime.deserialize_cuda_engine(recognizer_engine_data)
        self.recognizer_engine = recognizer_engine

        self.execution_context = recognizer_engine.create_execution_context()
        self.stream = cuda.Stream()

        self.ALPHABET = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.БГДЗИЛӨПУЦЧЭЯ"
        self.ALPHABET_SIZE = len(self.ALPHABET)
        self.SEQUENCE_SIZE = 30
        self.BLANK_INDEX = 0
        self.IMG_WIDTH = 128
        self.IMG_HEIGHT = 32
        self.PIXEL_MAX_VALUE = 255.0

        self.input_size = 1 * 3 * 32 * 128
        self.lp_size = 1 * 30 * 51
        self.country_size = 1 * 13

        self.flattened_input = np.empty(self.input_size, dtype=np.float32)
        self.lp_predictions = np.empty(self.lp_size, dtype=np.float32)
        self.country_predictions = np.empty(self.country_size, dtype=np.float32)

        self.cuda_buffer = [cuda.mem_alloc(self.flattened_input.nbytes),
                            cuda.mem_alloc(self.lp_predictions.nbytes),
                            cuda.mem_alloc(self.country_predictions.nbytes)]

    def prepare_image(self, frame):
        input_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        input_frame = cv2.resize(input_frame, (self.IMG_WIDTH, self.IMG_HEIGHT))
        input_frame = input_frame.transpose((2, 0, 1)).astype(np.float32) / self.PIXEL_MAX_VALUE
        return np.expand_dims(input_frame, 0)
    
    def execute_engine(self, frame):
        img = self.prepare_image(frame).flatten()

        cuda.memcpy_htod_async(self.cuda_buffer[0], img, self.stream)

        for i in range(self.recognizer_engine.num_io_tensors):
            self.execution_context.set_tensor_address(self.recognizer_engine.get_tensor_name(i), self.cuda_buffer[i])
        
        self.execution_context.execute_async_v3(stream_handle=self.stream.handle)

        #self.execution_context.execute_async(bindings=[int(buf) for buf in self.cuda_buffer],
        #                                     stream_handle=self.stream.handle)
        
        self.stream.synchronize()
        
        cuda.memcpy_dtoh_async(self.lp_predictions, self.cuda_buffer[1], self.stream)
        cuda.memcpy_dtoh_async(self.country_predictions, self.cuda_buffer[2], self.stream)

        self.stream.synchronize()

        predictions = self.lp_predictions.reshape(1, 30, 51)
        country_predictions = self.country_predictions.reshape(1, 13)

        return predictions, country_predictions

    def get_country_code_from_prediction(self, country_predictions):
        max_prob_index = country_predictions.argmax().item()
        max_prediction = country_predictions.max().item()
        return INDEX_TO_COUNTRY.get(max_prob_index, "UNIDENTIFIED_COUNTRY"), max_prediction

    def predict(self, frame):
        predictions, country_predictions = self.execute_engine(frame)
        predictions = predictions.squeeze(0)
        
        prob = 1.0
        current_label = []
        current_char = self.BLANK_INDEX
        current_prob = 1.0

        for i in range(self.SEQUENCE_SIZE):
            max_prob = 0.0
            max_index = 0

            for j in range(self.ALPHABET_SIZE):
                if max_prob < predictions[i, j]:
                    max_index = j
                    max_prob = predictions[i, j]

            if max_index == current_char:
                current_prob = max(max_prob, current_prob)
            else:
                if current_char != self.BLANK_INDEX:
                    current_label.append(self.ALPHABET[current_char])
                    prob *= current_prob
                current_prob = max_prob
                current_char = max_index

        if current_char != self.BLANK_INDEX:
            current_label.append(self.ALPHABET[current_char])
            prob *= current_prob

        if not current_label:
            current_label.append(self.ALPHABET[self.BLANK_INDEX])
            prob = 0.0

        car_label = ''.join(current_label)
        country_code, country_prob = self.get_country_code_from_prediction(country_predictions)

        return {
            'label': car_label,
            'country_code': country_code,
            'prob': prob,
            'country_prob': country_prob
        }
    