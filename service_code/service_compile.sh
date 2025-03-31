# cd /home/alikhan/Desktop/projects/project/service_code/lane_lp_handler

# python3 buildEngine.py \
#  --models_directory ./ \
#  --onnx_lane_file_name onnx_holder/nano_fused_removed_960_lane_surface_modified_slim_dynamic4_slim2r_slim_t.onnx \
#  --onnx_recognizer_file_name onnx_holder/recognizer.onnx \
#  --onnx_craft_file_name onnx_holder/craft.onnx \
#  --onnx_yolo_file_name onnx_holder/fused_removed_960_car_model_14_modified_slim_dynamic2.onnx


# torch-model-archiver --model-name laneLpService \
# --version 0.1 \
# --handler ./laneLpHandler.py \
# --serialized-file ./engine_holder/nano_fused_removed_960_lane_surface_modified_slim_dynamic4_slim2r_slim_t.engine \
# --export-path '/home/alikhan/Desktop/projects/project/service_holder' \
# --config-file ./model-config.yaml \
# --extra-files "./engine_holder/recognizer.engine","./engine_holder/craft.engine","./engine_holder/fused_removed_960_car_model_14_modified_slim_dynamic2.engine","./lpSrc/" --force

#!/bin/bash
cd /home/alikhan/Desktop/projects/project/service_code/lane_lp_handler

python3 buildEngine.py \
 --models_directory ./ \
 --onnx_lane_file_name onnx_holder/nano_fused_removed_960_lane_surface_modified_slim_dynamic4_slim2r_slim_t.onnx \
 --onnx_recognizer_file_name onnx_holder/recognizer.onnx \
 --onnx_craft_file_name onnx_holder/craft.onnx \
 --onnx_yolo_file_name onnx_holder/fused_removed_960_car_model_14_modified_slim_dynamic2.onnx

# Create the lane and license plate service model archive
torch-model-archiver --model-name laneLpService \
--version 0.1 \
--handler ./laneLpHandler.py \
--serialized-file ./engine_holder/nano_fused_removed_960_lane_surface_modified_slim_dynamic4_slim2r_slim_t.engine \
--export-path '/home/alikhan/Desktop/projects/project/service_holder' \
--config-file ./model-config.yaml \
--extra-files "./engine_holder/recognizer.engine","./engine_holder/craft.engine","./engine_holder/fused_removed_960_car_model_14_modified_slim_dynamic2.engine","./lpSrc/" --force

# Create the car detection service model archive
torch-model-archiver --model-name carDetectService \
--version 0.1 \
--handler ./carDetectHandler.py \
--serialized-file ./engine_holder/fused_removed_960_car_model_14_modified_slim_dynamic2.engine \
--export-path '/home/alikhan/Desktop/projects/project/service_holder' \
--config-file ./model-config.yaml \
--extra-files "./lpSrc/" --force
