import os
import sys
import json
import logging
import base64
import cv2
import io

from PIL import Image
from timeit import default_timer as timer
from ts.torch_handler.base_handler import BaseHandler
from functools import wraps

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

logger_base = logging.getLogger(__name__)
logger_base.info(trt.__version__)

assert trt.Builder(trt.Logger())

class CarDetectHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.model_state = "not_started"
        self.engine_path = ""
        self.batch_size = 1
        self.debug_model = True
        self.iou_threshold = 0.8
        self.score_threshold = 0.3
        self.class_names = ["Car", "License Plate", "327 sign", "328 sign"]
    
    def load_onnx_model(self, onnx_model_path):
        with open(onnx_model_path, 'rb') as f:
            onnx_model = f.read()
        return onnx_model
    
    def yolo_model_build(self, models_directory, onnx_file_name):
        onnx_path = os.path.join(models_directory, onnx_file_name)
        if not os.path.isfile(onnx_path):
            raise RuntimeError(f"Missing ONNX model file: {onnx_path}")
        
        self.engine_path = os.path.join(models_directory, "engine_holder", os.path.basename(onnx_path).split('.')[0] + ".engine")
        if os.path.isfile(self.engine_path):
            self.model_state = "builded"
            logger_base.info(f"Car Engine File Already Exists!")
            print(f"Car Engine File Already Exists!")
            return
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        if not parser.parse(self.load_onnx_model(onnx_path)):
            for error in range(parser.num_errors):
                logger_base.info(parser.get_error(error))
            raise RuntimeError("ONNX parsing broken")
        
        logger_base.info(f"Parsed ONNX!")
        print(f"Parsed ONNX!")
        
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        
        # optimization profile for dynamic batching
        profile = builder.create_optimization_profile()
        profile.set_shape('pre-preop-input', 
                    (1, 544, 960, 3),   # min shape
                    (4, 544, 960, 3),   # optimal shape 
                    (8, 544, 960, 3))   # max shape
        config.add_optimization_profile(profile)
        
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("Failed to build the engine!")
        
        logger_base.info(f"Built engine!")
        print(f"Built engine!")

        with open(self.engine_path, "wb") as f:
            f.write(engine)
        
        logger_base.info(f"Engine File Built And Saved!")
        print(f"Engine File Built And Saved!")
        self.model_state = "builded"
    
    def read_engine(self):
        if not os.path.isfile(self.engine_path):
            raise RuntimeError(f"Missing Engine Model File: {self.engine_path}")
        with open(self.engine_path, "rb") as f:
            engine = f.read()
        self.model_state = "engine_read"
        logger_base.info(f"Engine is Found and Read!")
        
        return engine
    
    def allocate_buffers(self, engine):
        '''
        Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
        '''
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                size = trt.volume(engine.get_tensor_shape(tensor_name)[1:]) * self.batch_size
            else:
                # Output is BatchSize*200, 9 for detections
                size = trt.volume([self.batch_size * 200, 9])
                
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to device bindings
            bindings.append(int(device_mem))

            # Append to the appropriate input/output list
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append((host_mem, device_mem))
            else:
                outputs.append((host_mem, device_mem))

        return inputs, outputs, bindings, stream
    
    def base_initialize(self, context):
        # Define input/output dimensions
        self.orig_w = 1920
        self.orig_h = 1080
        self.input_w = 960
        self.input_h = 544
        
        self.manifest = context.manifest
        properties = context.system_properties
        
        models_directory = properties.get("model_dir")
        engine_file = self.manifest['model']['serializedFile']
        self.engine_path = os.path.join(models_directory, engine_file)
        
        engine = self.read_engine()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine)
        self.context = self.engine.create_execution_context()
        
        logger_base.info(f"Engine is Deserialized!")
        
        inputs, outputs, bindings, stream = self.allocate_buffers(self.engine)
        logger_base.info("Buffers for Engine Run Allocated!")
        
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream
        self.model_state = "allocated"
    
    def initialize(self, context):
        try:
            self.base_initialize(context)
        except Exception as excp:
            logger_base.info(f"TensorRT model initialization failed: {str(excp)}")
            raise excp
    
    def read_json(self, request):
        inputs = []
        unique_names = []
        images_sizes = []
        date_times = []
        image_mat_debug = None
        
        for sub_request in request:
            bodik = sub_request.get("body")
            
            if isinstance(bodik, dict):
                request_body_json = bodik
            else:
                request_body_json = json.loads(bodik)
            
            unique_names.append(request_body_json['unique_id'])
            date_times.append(request_body_json['dateTime'])
            
            # Decode image
            image_data = base64.b64decode(request_body_json['generalFrame'])
            start_time = timer()
            image_mat = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            json_read_time = round(timer() - start_time, 4)
            logger_base.info(f'From buffer time: {str(json_read_time)} s.')
            
            if self.debug_model:
                image_mat_debug = image_mat.copy()
            
            # Save original image dimensions
            images_sizes.append(image_mat.shape)
            
            # Preprocess the image
            image_mat = cv2.cvtColor(image_mat, cv2.COLOR_BGR2RGB)
            image_mat = cv2.resize(image_mat, (self.input_w, self.input_h))
            image_mat = image_mat.astype(np.float32)
            image_mat = image_mat.ravel()  # Flatten the array
            inputs.append(image_mat)
        
        # Return the last processed image and metadata
        return inputs[-1], image_mat_debug, unique_names[-1], images_sizes[-1], date_times[-1]
    
    def do_inference(self, flattened_image):
        try:
            np.copyto(self.inputs[0][0], flattened_image)
            
            # Copy from host to device
            [cuda.memcpy_htod_async(inp[1], inp[0], self.stream) for inp in self.inputs]
            
            # Set zeros in output buffer to avoid stale data
            cuda.memset_d32(self.outputs[0][1], 0, self.outputs[0][0].nbytes // 4)
            
            # Set input shape based on batch_size
            self.context.set_input_shape(self.engine.get_tensor_name(0), (self.batch_size, self.input_h, self.input_w, 3))
            
            # Set tensor addresses
            for i in range(self.engine.num_io_tensors):
                self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
                
            # Run inference
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            
            # Synchronize the stream
            self.stream.synchronize()
            
            # Copy from device to host
            [cuda.memcpy_dtoh_async(out[0], out[1], self.stream) for out in self.outputs]
            
            # Another synchronization to ensure memory transfers are done
            self.stream.synchronize()
            
        except Exception as excp:
            logger_base.info(f"TensorRT model inference failed: {str(excp)}")
            raise excp

        return [out[0] for out in self.outputs]
    
    def rescale_coordinates(self, coords, orig_height, orig_width):
        """
        Rescale coordinates from the model's output space (1920x1080) to the original image size
        """
        x1, y1, x2, y2 = coords
        
        # Scale coordinates from 1920x1080 to the original image size
        x1 = x1 / self.orig_w * orig_width
        x2 = x2 / self.orig_w * orig_width
        y1 = y1 / self.orig_h * orig_height
        y2 = y2 / self.orig_h * orig_height
        
        # Clip coordinates to image boundaries
        x1 = np.clip(x1, 0, orig_width)
        x2 = np.clip(x2, 0, orig_width)
        y1 = np.clip(y1, 0, orig_height)
        y2 = np.clip(y2, 0, orig_height)
        
        return [x1, y1, x2, y2]
    
    def non_max_suppression(self, boxes, scores, classes, iou_threshold, score_threshold):
        """
        Apply non-maximum suppression to eliminate redundant overlapping boxes
        """
        # Convert boxes to format expected by cv2.dnn.NMSBoxes
        boxes_for_nms = []
        final_scores = []
        final_class_ids = []
        
        for i, (box, score_array, class_array) in enumerate(zip(boxes, scores, classes)):
            x1, y1, x2, y2 = box
            if max(score_array) > score_threshold:
                class_id = np.argmax(score_array)
                score = score_array[class_id]
                boxes_for_nms.append([x1, y1, x2 - x1, y2 - y1])  # Convert to [x, y, w, h]
                final_scores.append(float(score))
                final_class_ids.append(int(class_id))
        
        if not boxes_for_nms:
            return [], [], []
            
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, final_scores, score_threshold, iou_threshold)
        
        # Gather results after NMS
        nms_boxes = []
        nms_scores = []
        nms_class_ids = []
        
        for idx in indices:
            # In OpenCV 4.5.4+, idx is a scalar rather than a 1-element array
            idx = idx.item() if hasattr(idx, 'item') else idx
            box = boxes_for_nms[idx]
            # Convert back to [x1, y1, x2, y2] format
            nms_boxes.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
            nms_scores.append(final_scores[idx])
            nms_class_ids.append(final_class_ids[idx])
            
        return nms_boxes, nms_scores, nms_class_ids
    
    def process_detections(self, outputs, image_mat_debug, unique_name, images_sizes, date_time):
        """
        Process raw detections from the model
        """
        orig_height, orig_width, _ = images_sizes
        
        # Parse output tensor which is (batch_size*200, 9)
        detections_raw = outputs[0].reshape(-1, 9)
        
        # Separate arrays for batch_index, boxes, class scores for each detection
        batch_indices = detections_raw[:, 0].astype(np.int32)
        boxes = detections_raw[:, 1:5]  # x1, y1, x2, y2
        class_scores = detections_raw[:, 5:]  # class probabilities
        
        # Filter only valid detections for this batch
        valid_detections = batch_indices == 0  # For single image processing, batch index is 0
        filtered_boxes = boxes[valid_detections]
        filtered_class_scores = class_scores[valid_detections]
        
        # Rescale boxes to original image size
        rescaled_boxes = [self.rescale_coordinates(box, orig_height, orig_width) for box in filtered_boxes]
        
        # Apply additional NMS to handle potential model bug
        nms_boxes, nms_scores, nms_class_ids = self.non_max_suppression(
            rescaled_boxes, 
            filtered_class_scores, 
            np.argmax(filtered_class_scores, axis=1),
            self.iou_threshold, 
            self.score_threshold
        )
        
        # Create detection results list
        detections = []
        for box, score, class_id in zip(nms_boxes, nms_scores, nms_class_ids):
            x1, y1, x2, y2 = map(int, box)
            detection = {
                "bbox": [x1, y1, x2, y2],
                "class_id": int(class_id),
                "class_name": self.class_names[class_id],
                "confidence": float(score)
            }
            detections.append(detection)
            
            # Draw detections on debug image if enabled
            if self.debug_model and image_mat_debug is not None:
                color = (0, 255, 0)  # BGR Green
                thickness = 2
                cv2.rectangle(image_mat_debug, (x1, y1), (x2, y2), color, thickness)
                
                # Add text label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                text = f"{self.class_names[class_id]}: {score:.2f}"
                cv2.putText(image_mat_debug, text, (x1, y1 - 10), font, font_scale, color, thickness)
        
        # Save debug image if enabled
        if self.debug_model and image_mat_debug is not None:
            if not os.path.isdir("./save_folder"):
                os.makedirs("./save_folder")
            image_name = f"detect_{unique_name}_{date_time}.jpg"
            cv2.imwrite(os.path.join("./save_folder", image_name), image_mat_debug)
        
        # Create final response
        response = {
            "status_code": 200,
            "unique_id": unique_name,
            "detections": detections
        }
        
        return response
    
    def execute_and_count_time(self, function, args, log_time_name):
        @wraps(function)
        def wrapper(*args):
            start_time = timer()
            result = function(*args) if isinstance(args, tuple) else function(args)
            execute_time = round(timer() - start_time, 4)
            logger_base.info(f'{log_time_name} time: {str(execute_time)} s.')
            return {
                "result": result,
                "time": execute_time
            }
        return wrapper(*args) if isinstance(args, tuple) else wrapper(args)
    
    def handle(self, data, context):
        if data is None:
            logger_base.warning("No Input Data is provided!")
            return None
        
        try:
            # Read and parse the input data
            res = self.execute_and_count_time(
                self.read_json, data, 'Read'
            )
            flattened_image, image_mat_debug, unique_name, images_sizes, date_time = res['result']
            json_read_time = res['time']
            
            # Run inference
            res = self.execute_and_count_time(
                self.do_inference, flattened_image, 'Inference'
            )
            outputs = res['result']
            inference_time = res['time']
            
            # Process detections
            res = self.execute_and_count_time(
                self.process_detections, (outputs, image_mat_debug, unique_name, images_sizes, date_time), 'Post-process'
            )
            final_output = res['result']
            post_time = res['time']
            
            # Log total prediction time
            total_time = json_read_time + inference_time + post_time
            logger_base.info(f'Total prediction time: {str(total_time)} s.')
            
        except Exception as excp:
            logger_base.info(f"Car detection model inference failed: {str(excp)}")
            return [json.dumps(
                {
                    "status_code": 500,
                    "error_body": str(excp)
                }
            )]
        
        return [json.dumps(final_output)]