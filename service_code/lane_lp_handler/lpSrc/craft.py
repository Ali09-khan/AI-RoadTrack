import cv2
import torch
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from torch.autograd import Variable
from typing import List, Tuple, Any

from utils import (
    resize_aspect_ratio,
    normalizeMeanVariance,
    get_det_boxes,
    adjustResultCoordinates,
    cvt2HeatmapImg,
    crop_image,
    minimum_bounding_rectangle,
    distance,
    split_boxes,
    normalize_rect,
    addopt_rect_to_bbox,
    add_coordinates_offset,
    filter_boxes,
    make_rect_variants,
    get_cv_zone_rgb,
    reshape_points,
    detect_best_perspective,
    normalize_perspective_images
)


class NpPointsCraft(object):
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f:
            net_engine_data = f.read()

        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        net_engine = runtime.deserialize_cuda_engine(net_engine_data)
        self.net_engine = net_engine

        if net_engine is None:
            raise ValueError(
                "Failed to deserialize the engine. "
                "This might be due to a version mismatch between the TensorRT runtime "
                "and the version used to build the engine."
            )

        self.net_execution_context = net_engine.create_execution_context()
        self.stream = cuda.Stream()

        # max observed input/outputs sizes
        self.x_size = 1 * 3 * 192 * 320
        self.y_size = 1 * 96 * 160 * 2
        self.refiner_size = 1 * 1 * 96 * 160 * 1

        self.flattened_x = np.empty(self.x_size, dtype=np.float32)
        self.y_predictions = np.empty(self.y_size, dtype=np.float32)
        self.refiner_predictions = np.empty(self.refiner_size, dtype=np.float32)
        
        self.cuda_buffer = [cuda.mem_alloc(self.flattened_x.nbytes),
                            cuda.mem_alloc(self.refiner_predictions.nbytes),
                            cuda.mem_alloc(self.y_predictions.nbytes),
                            ]


    @staticmethod
    def preprocessing_craft(image, canvas_size, mag_ratio):
        # resize
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
            image,
            canvas_size,
            interpolation=cv2.INTER_LINEAR)
        ratio_h = ratio_w = 1 / target_ratio
        x = normalizeMeanVariance(img_resized)
        return x, ratio_h, ratio_w
    


    @staticmethod
    def craft_postprocessing(score_text: np.ndarray, score_link: np.ndarray, text_threshold: float,
                             link_threshold: float, low_text: float, ratio_w: float, ratio_h: float):
        # Post-processing
        boxes = get_det_boxes(score_text, score_link, text_threshold, link_threshold, low_text)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = cvt2HeatmapImg(render_img)
        return boxes, ret_score_text
    


    @torch.no_grad()
    def forward(self, x: np.ndarray) -> Tuple[Any, Any]:
        """
        TODO: describe function
        """
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]

        x_shape = x.shape
        x = x.flatten()
            
        cuda.memcpy_htod_async(self.cuda_buffer[0], x.numpy(), self.stream)

        self.net_execution_context.set_input_shape(self.net_engine.get_tensor_name(0), (1, 3, x_shape[2], 320))
        
        for i in range(self.net_engine.num_io_tensors):
            self.net_execution_context.set_tensor_address(self.net_engine.get_tensor_name(i), self.cuda_buffer[i])
        
        self.net_execution_context.execute_async_v3(stream_handle=self.stream.handle)
        
        #self.net_execution_context.execute_async(bindings=[int(buf) for buf in self.cuda_buffer],
        #                                     stream_handle=self.stream.handle)
        self.stream.synchronize()
        
        #cuda.memcpy_dtoh_async(self.y_predictions, self.cuda_buffer[1], self.stream)
        #cuda.memcpy_dtoh_async(self.refiner_predictions, self.cuda_buffer[2], self.stream)

        cuda.memcpy_dtoh_async(self.y_predictions, self.cuda_buffer[2], self.stream)
        cuda.memcpy_dtoh_async(self.refiner_predictions, self.cuda_buffer[1], self.stream)

        self.stream.synchronize()

        y = self.y_predictions.reshape(1, -1, 160, 2)[:, :x_shape[2]//2, :, :].copy()
        refiner = self.refiner_predictions.reshape(1, 1, -1, 160, 1)[:, :, :x_shape[2]//2, :, :].copy()

        score_text = y[0, :, :, 0]
        score_link = refiner[0, 0, :, :, 0]

        return score_text, score_link


    def detect(self,
               inputs,
               canvas_size: int = 300,
               mag_ratio: float = 1.0,
               quality_profile: List = None,
               text_threshold: float = 0.6,
               link_threshold: float = 0.7,
               low_text: float = 0.4
               ):
        preprocessed_data = self.preprocess(inputs, canvas_size, mag_ratio)
        model_outputs = self.forward_batch(preprocessed_data)
        return self.postprocess(model_outputs, quality_profile, text_threshold, link_threshold, low_text)



    @torch.no_grad()
    def forward_batch(self, inputs: Any, **_) -> Any:
        return [[*self.forward(x[0]), *x[1:]] for x in inputs]



    def preprocess(self, inputs: Any, canvas_size: int = 300, mag_ratio: float = 1.0, **_) -> Any:
        res = []
        for image_id, (image, target_boxes) in enumerate(inputs):
            for target_box in target_boxes:
                image_part, (x0, w0, y0, h0) = crop_image(image, target_box)
                if h0 / w0 > 3.5:
                    image_part = cv2.rotate(image_part, cv2.ROTATE_90_CLOCKWISE)
                x, ratio_h, ratio_w = self.preprocessing_craft(image_part, canvas_size, mag_ratio)
                res.append([x, image, ratio_h, ratio_w, target_box, image_id, (x0, w0, y0, h0), image_part])
        return res

   
    
    def postprocess(self, inputs: Any,
                    quality_profile: List = None,
                    text_threshold: float = 0.6,
                    link_threshold: float = 0.7,
                    low_text: float = 0.4,
                    in_zone_only: bool = False,
                    **_) -> Any:
        if quality_profile is None:
            quality_profile = [1, 0, 0, 0]

        all_points = []
        all_mline_boxes = []
        all_image_ids = []
        all_count_lines = []
        all_image_parts = []
        for score_text, score_link, image, ratio_h, ratio_w, target_box, image_id, (x0, w0, y0, h0), image_part \
                in inputs:
            all_image_parts.append(image_part)
            image_shape = image_part.shape
            all_image_ids.append(image_id)
            bboxes, ret_score_text = self.craft_postprocessing(
                score_text, score_link, text_threshold,
                link_threshold, low_text, ratio_w, ratio_h)
            dimensions = [{'dx': distance(poly[0], poly[1]), 'dy': distance(poly[1], poly[2])}
                          for poly in bboxes]
            np_bboxes_idx, garbage_bboxes_idx = split_boxes(bboxes, dimensions)
            multiline_rects = [bboxes[i] for i in np_bboxes_idx]

            probably_count_lines = 1
            target_points = []
            if len(np_bboxes_idx) == 1:
                target_points = bboxes[np_bboxes_idx[0]]
            if len(np_bboxes_idx) > 1:
                started_boxes = np.concatenate([bboxes[i] for i in np_bboxes_idx], axis=0)
                target_points = minimum_bounding_rectangle(np.concatenate(multiline_rects, axis=0))
                np_bboxes_idx, garbage_bboxes_idx, probably_count_lines = filter_boxes(bboxes, dimensions,
                                                                                       target_points, np_bboxes_idx)
                filtred_boxes = np.concatenate([bboxes[i] for i in np_bboxes_idx], axis=0)
                if len(started_boxes) > len(filtred_boxes):
                    target_points = minimum_bounding_rectangle(started_boxes)
            if len(np_bboxes_idx) > 0:
                target_points = normalize_rect(target_points)
                target_points = addopt_rect_to_bbox(target_points, image_shape, 7, 12, 0, 12)
            all_count_lines.append(probably_count_lines)

            local_propably_points, mline_boxes = target_points, multiline_rects
            all_mline_boxes.append(mline_boxes)
            propably_points = add_coordinates_offset(local_propably_points, x0, y0)
            if len(propably_points):
                target_points_variants = make_rect_variants(propably_points, quality_profile)
                if len(target_points_variants):
                    target_points_variants = make_rect_variants(propably_points, quality_profile)
                    if len(target_points_variants) > 1:
                        img_parts = [get_cv_zone_rgb(image, reshape_points(rect, 1)) for rect in target_points_variants]
                        idx = detect_best_perspective(normalize_perspective_images(img_parts))
                        points = target_points_variants[idx]
                    else:
                        points = target_points_variants[0]
                    if in_zone_only:
                        for i in range(len(points)):
                            points[i][0] = x0 if points[i][0] < x0 else points[i][0]
                            points[i][1] = y0 if points[i][1] < y0 else points[i][1]
                            points[i][0] = x0 + w0 if points[i][0] > x0 + w0 else points[i][0]
                            points[i][1] = y0 + h0 if points[i][1] > y0 + h0 else points[i][1]
                    all_points.append(points)
                else:
                    all_points.append([
                        [x0, y0 + h0],
                        [x0, y0],
                        [x0 + w0, y0],
                        [x0 + w0, y0 + h0]
                    ])
        if len(all_image_ids):
            n = max(all_image_ids) + 1
        else:
            n = 1
        images_points = [[] for _ in range(n)]
        images_mline_boxes = [[] for _ in range(n)]
        for point, mline_box, image_id in zip(all_points, all_mline_boxes, all_image_ids):
            images_points[image_id].append(point)
            images_mline_boxes[image_id].append(mline_box)
        return images_points, images_mline_boxes