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
import pycuda.autoinit # ?


try:
    from init_and_run import start_model, inference
except ModuleNotFoundError:
    print("Warning: init_and_run is not used here. Continuing without it.")
    #from lpSrc.init_and_run import start_model, inference
##### Start #####
#################

logger_base = logging.getLogger(__name__)
logger_base.info(trt.__version__)

assert trt.Builder(trt.Logger())

class LaneLpHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        #self.initialized = False
        self.model_state = "not_started"
        #self.onnx_path = ""
        self.engine_path = ""
        self.lp_path = ""
        self.craft_engine_path = ""
        self.recognizer_engine_path = ""
        self.batch_size = 1
        self.debug_model = True
        self.lp_proccesing = True
    
    def load_onnx_model(self, onnx_model_path):
        with open(onnx_model_path, 'rb') as f:
            onnx_model = f.read()
        return onnx_model
    
    def lane_model_build(self, models_directory, onnx_file_name):
        onnx_path = os.path.join(models_directory, onnx_file_name)
        if not os.path.isfile(onnx_path):
            raise RuntimeError("Missing onnx model")
        #
        self.engine_path = os.path.join(models_directory, "engine_holder", os.path.basename(onnx_path).split('.')[0] + ".engine")
        if os.path.isfile(self.engine_path):
            self.model_state = "builded"
            logger_base.info(f"Lane Engine File Already Exist!")
            print(f"Lane Engine File Already Exist!")
            return
        #
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        #
        if not parser.parse(self.load_onnx_model(onnx_path)):
            for error in range(parser.num_errors):
                logger_base.info(parser.get_error(error))
            raise RuntimeError("onnx parsing broken")
        #
        logger_base.info(f"Parsed ONNX!")
        print(f"Parsed ONNX!")
        
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        profile.set_shape('pre-preop-input', 
                    (self.batch_size, 544, 960, 3), 
                    (self.batch_size, 544, 960, 3), 
                    (self.batch_size, 544, 960, 3))
        config.add_optimization_profile(profile)
        #
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("Failed to build the engine!")
        #
        
        logger_base.info(f"Builded engine!")
        print(f"Builded engine!")

        with open(self.engine_path, "wb") as f:
            f.write(engine)
        
        logger_base.info(f"Lane Engine File Built And Saved!")
        print(f"Lane Engine File Built And Saved!")
        self.model_state = "builded"
    
    def read_engine(self):
        if not os.path.isfile(self.engine_path):
            raise RuntimeError("Missing Engine Model File")
        with open(self.engine_path, "rb") as f:
            engine = f.read()
        self.model_state = "engine_read"
        logger_base.info(f"Engine is Found and Read!")
        
        return engine
    
    def read_lp_engine(self, craft_engine_path, recognizer_engine_path):
        craft_model, recognizer_model = start_model(craft_engine_path, recognizer_engine_path)
        return craft_model, recognizer_model
    
    
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
                if engine.get_tensor_shape(tensor_name)[-1] == 9:
                    size = trt.volume([30, 9]) * self.batch_size
                else:
                    size = trt.volume(engine.get_tensor_shape(tensor_name)[1:]) * self.batch_size
                
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype) # page-locked memory buffer (won't swapped to disk) 
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to device bindings. 
            # When cast to int, it's a linear index into the context's memory (like memory address). 
            bindings.append(int(device_mem))


            # Append to the appropriate input/output list.
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append((host_mem, device_mem))
            else:
                outputs.append((host_mem, device_mem))

        return inputs, outputs, bindings, stream
    
    
    def base_initialize(self, context):
        # batch can be defined here
        #
        self.w = 1920
        self.h = 1080
        #
        self.input_w = 960
        self.input_h = 544
        #
        #self.allowed_slope_ranges = [
        #[0.05, 101], #    [0.25, 101],
        #[-101, -0.05] #    [-101, -0.25]
        #]
        self.allowed_slope_ranges = {
            'front': [
                [0.05, 101],
            ],
            'rear': [
                [-101, -0.05],
            ]
        }
        #
        self.manifest = context.manifest
        properties = context.system_properties
        #
        models_directory = properties.get("model_dir")
        engine_file = self.manifest['model']['serializedFile']
        self.engine_path = os.path.join(models_directory, engine_file)
        #
        #self.lane_model_build(models_directory)
        engine = self.read_engine()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine)
        self.context = self.engine.create_execution_context()
        
        logger_base.info(f"Engine is Deserialized!")
        
        ### Here could be efficiency problems ###
        inputs, outputs, bindings, stream = self.allocate_buffers(self.engine)
        logger_base.info("Buffers for Engine Run Allocated!")
        #########################################
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream
        self.model_state = "allocated"
        #### Lane engine or onnx read ###
        if self.lp_proccesing:
            self.craft_engine_path = os.path.join(models_directory, "craft.engine")
            self.recognizer_engine_path = os.path.join(models_directory, "recognizer.engine")
            self.craft, self.recognizer = self.read_lp_engine(self.craft_engine_path, 
                                                                     self.recognizer_engine_path)
            
        #################################
        
    def initialize(self, context):
        try:
            self.base_initialize(context)
        except Exception as excp:
            # Log and re-raise the exception
            logger_base.info(f"TensoRT model initialization failed: {str(excp)}")
            raise excp
    
    def read_json(self, request):
        inputs = []
        inputs_lp = [None]
        directions = []
        unique_names = []
        enable_filtering = []
        images_sizes = []
        date_times = []
        image_mat_debug = None
        
        #logger_base.info(f"Input came: {request}")
        
        for sub_request in request:
            bodik = sub_request.get("body")
            #logger_base.info(f"Input proccesed: {bodik}")
            
            if isinstance(bodik, dict):
                request_body_json = bodik
            else:
                request_body_json = json.loads(bodik)
                
            #logger_base.info(f"Input load: {request_body_json}")
            
            
            unique_names.append(request_body_json['unique_id'])
            date_times.append(request_body_json['dateTime'])
            directions.append(request_body_json['direction'])
            enable_filtering.append(True if request_body_json['enable_filtering'] == 'enable' else False)
            
            #
            image_data = base64.b64decode(request_body_json['generalFrame'])
            start_time = timer()
            image_mat = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            json_read_time = round(timer() - start_time, 4)
            logger_base.info(f'From buffer time: {str(json_read_time)} s.')
            #
            if self.lp_proccesing:
                # for efficiency will do it like this for now
                if 'lp_Coordinates' in request_body_json:
                    general_frame_image_mat = image_mat.copy()
            #
            if self.debug_model:
                image_mat_debug = image_mat.copy()
            #
            image_mat = cv2.cvtColor(image_mat, cv2.COLOR_BGR2RGB)
            images_sizes.append(image_mat.shape)
            #
            
            image_mat = cv2.resize(image_mat, (self.w, self.h))
            image_mat = cv2.resize(image_mat, (self.input_w, self.input_h))
            image_mat = image_mat.astype(np.float32)
            image_mat = image_mat.ravel()
            inputs.append(image_mat)
            #
            
            if self.lp_proccesing:
                if 'lp_Coordinates' in request_body_json:
                    input_lp = {
                        "general_frame": general_frame_image_mat,
                        'lp_Coordinates': request_body_json['lp_Coordinates']
                    }
                    inputs_lp.append(input_lp)
            
        # for now return just index 0
        return inputs[-1], image_mat_debug, directions[-1], inputs_lp[-1], \
            unique_names[-1], enable_filtering[-1], images_sizes[-1], date_times[-1]
    
    
    def do_inference(self, flattened_image):
        try:
            np.copyto(self.inputs[0][0], flattened_image)
            #
            [cuda.memcpy_htod_async(inp[1], inp[0], self.stream) for inp in self.inputs]
            # Dangerous code !!!
            cuda.memset_d32(self.outputs[0][1], 0, self.outputs[0][0].nbytes // 4)
            #
            self.context.set_input_shape(self.engine.get_tensor_name(0), (self.batch_size, 544, 960, 3))
            for i in range(self.engine.num_io_tensors):
                self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
            # Run inference
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            # Synchronize the stream
            self.stream.synchronize()
            
            #size = trt.volume([30, 9]) * self.batch_size
            #dtype = trt.nptype(self.engine.get_tensor_dtype(self.engine.get_tensor_name(0)))
            #host_mem = cuda.pagelocked_empty(size, dtype) # page-locked memory buffer (won't swapped to disk) 
            
            [cuda.memcpy_dtoh_async(out[0], out[1], self.stream) for out in self.outputs]
            #cuda.memcpy_dtoh_async(host_mem, self.outputs[0][1], self.stream)
            
        except Exception as excp:
            logger_base.info(f"TensoRT model inference failed: {str(excp)}")
            raise excp

        return [out[0] for out in self.outputs]
        #return [host_mem]
    
    
    def find_slope_and_x(self, x1, y1, x2, y2, h):
        
        if not ((x2-x1) == 0 or (y2-y1) == 0): 
            a = (y2-y1) / (x2-x1)
            b = y1 - a * x1
            y = h
            x = (y - b) / a
        elif  x2 == x1:
            x = x2
            a = 102 #"vertical"
        else:
            y = h
            x = x2
            a = 0 #"horizontal"
        return a, x, b
    
    def find_image_end_points(self, m, b, h, w):
        # Check intersection with the left edge (x = 0)
        edges = []
        y_at_left = m * 0 + b
        if 0 <= y_at_left <= h:
            edges.append((0, y_at_left))
        else:
            x_at_left = (0 - b) / m
            edges.append((x_at_left, 0))
            
        y_at_right = m * w + b
        if 0 <= y_at_right <= h:
            edges.append((w, y_at_right))
        else:
            x_at_right = (h - b) / m
            edges.append((x_at_right, h))
        
        return edges
    
    
    def check_slope(self, slope, direction):
        for yrange in self.allowed_slope_ranges[direction]:
            if slope > yrange[0] and slope < yrange[1]:
                return True
        return False
    
    def rescale_predictions(self, input_array, orig_height, orig_width):
        x1, y1, x2, y2 = input_array
        #
        x1 = x1 / self.w * orig_width
        x2 = x2 / self.w * orig_width
        #
        y1 = y1 / self.h * orig_height
        y2 = y2 / self.h * orig_height
        #
        x1 =  np.clip(x1, 0, orig_width)
        x2 =  np.clip(x2, 0, orig_width)
        #
        y1 =  np.clip(y1, 0, orig_height)
        y2 =  np.clip(y2, 0, orig_height)
        
        return [x1, y1, x2, y2]
    
    def do_inference_lp(self, inputs_lp, images_sizes):
        input_remade = {
            "general_frame":inputs_lp["general_frame"],
            "bbox": inputs_lp["lp_Coordinates"]
        }
        lp_prediction = inference(self.craft, self.recognizer, input_remade)
        return lp_prediction

    
    def lane_proccess(self, outputs, image_mat_debug, direction, 
                      unique_names, enable_filtering, images_sizes, date_times):
        #
        predicted_lanes = []
        x1, y1, x2, y2 = 0, 0, 0, 0
        orig_height, orig_width, _ = images_sizes
        #print(outputs[0])
        for i in range(0, self.batch_size*30*9, 9):
            inter_output = outputs[0][i:i+9]
            #
            inter_output[1:5] = self.rescale_predictions(inter_output[1:5], orig_height, orig_width)
            #
            if np.all(np.equal(inter_output[1:5], np.array([x1, y1, x2, y2]))):
                break
            #
            x1, y1, x2, y2 = inter_output[1:5]
            #
            if np.all(np.equal(inter_output[1:5], np.array([0, 0, 0, 0]))):
                continue
            
            class_1, class_2 = inter_output[5:7]
            sum1, sum2 = inter_output[7:9]
            
            if sum2 > sum1:
                out = [x1, y1, x2, y2]
            else:
                out = [x2, y1, x1, y2]
            
            slope, x_intersection, b = self.find_slope_and_x(*(out+[orig_height]))
            
            if enable_filtering:
                if not self.check_slope(slope, direction):
                    continue
            
            edges = self.find_image_end_points(slope, b, orig_height, orig_width)
            edges = sorted(edges, key=lambda d: d[1])
            
            if class_1 > class_2:
                class_type = "edge"
            else:
                class_type = "line"
            
            predicted_lanes.append(
                    {
                    "start": [int(x) for x in out[:2]], 
                    "end": [int(x) for x in out[2:]], 
                    "image_start": [int(x) for x in edges[0]], 
                    "image_end": [int(x) for x in edges[1]], 
                    "type": class_type,
                    "x_intersection": int(x_intersection),
                    "slope": eval(str(round(slope, 4))),
                    "b": eval(str(round(b, 4)))
                    }
                
            )
        if len(predicted_lanes) > 0:
            if direction == 'front':
                reversing = True
            else:
                reversing = False
            predicted_lanes = sorted(predicted_lanes, key=lambda d: d['x_intersection'], reverse=reversing)
        
        i = 0
        for out_index, pred in enumerate(predicted_lanes):
            predicted_lanes[out_index]['index'] = i
            i = i + 1
            if self.debug_model:
                if pred['type'] == 'line':
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                thickness = 5
                fontScale = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                cv2.line(image_mat_debug, pred['start'], pred['end'], color, 7)
                cv2.line(image_mat_debug, pred['image_start'], pred['image_end'], (0, 0, 0), 2)
                image_mat_debug = cv2.putText(image_mat_debug, str(pred["slope"]), pred['start'], font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        
        if self.debug_model:
            if not os.path.isdir("./save_folder"):
                os.makedirs("./save_folder")
            image_name = direction + "_" + str(date_times) + '.jpg'
            cv2.imwrite(os.path.join("./save_folder", image_name), image_mat_debug)
        
        final_output = {
            "status_code": 200,
            "lines": predicted_lanes,
        }
        
        return final_output
    
    def post_proccess_lp(self, final_output, unique_name, lp):
        final_output['unique_id'] = unique_name
        final_output['lp'] = lp
        return final_output
    
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
            res = self.execute_and_count_time(
                self.read_json, data, 'Read'
            )
            flattened_image, image_mat_debug, direction, inputs_lp, \
            unique_names, enable_filtering, images_sizes, date_times = res['result']
            json_read_time = res['time']
            #
            res = self.execute_and_count_time(
                self.do_inference, flattened_image, 'Enq-Lane'
            )
            outputs = res['result']
            enq_time = res['time']
            #
            res = self.execute_and_count_time(
                self.lane_proccess, (outputs, image_mat_debug, direction, 
                                     unique_names, enable_filtering, images_sizes, date_times
                                     ), 'Post-Lane'
            )
            final_output = res['result']
            post_time = res['time']
            #
            if self.lp_proccesing and inputs_lp is not None:
                res = self.execute_and_count_time(
                self.do_inference_lp, (inputs_lp, images_sizes), 'Enq-LP'
                )
                lp = res['result']
                lp_time = res['time']
            else:
                lp = '00000000'
                lp_time = 0
            #
            final_output = self.post_proccess_lp(final_output, unique_names, lp)
            #
            count_time = json_read_time + enq_time + post_time + lp_time
            logger_base.info(f'Predict time: {str(count_time)} s.')
        
        except Exception as excp:
            logger_base.info(f"Lane model inference failed {str(excp)}")
            return [json.dumps(
                        {
                        "status_code": 228,
                        "error_body": str(excp)
                        }
                    )]
        
        return [json.dumps(final_output)]
