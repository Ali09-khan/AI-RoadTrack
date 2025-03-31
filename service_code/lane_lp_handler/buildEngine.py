# from laneLpHandler import LaneLpHandler
# import argparse
# from lpSrc.build import build_engines

# def main(models_directory, onnx_lane_file_name, onnx_recognizer_file_name, onnx_craft_file_name, onnx_yolo_file_name):
#     lanelp_builder = LaneLpHandler()
#     lanelp_builder.lane_model_build(models_directory, onnx_lane_file_name)
#     _, _ = build_engines(models_directory, onnx_craft_file_name, onnx_recognizer_file_name)
    

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--models_directory', 
#                         type=str, 
#                         required=True,
#                         help='path where onnx is stored and where engine will be saved')
#     parser.add_argument('--onnx_lane_file_name', 
#                         type=str, 
#                         required=True,
#                         help='Onnx lane file name')
#     parser.add_argument('--onnx_recognizer_file_name', 
#                         type=str, 
#                         required=True,
#                         help='Onnx recognizer file name')
#     parser.add_argument('--onnx_craft_file_name', 
#                         type=str, 
#                         required=True,
#                         help='Onnx craft file name')
#     parser.add_argument('--onnx_yolo_file_name', 
#                     type=str, 
#                     default='onnx_holder/fused_removed_960_car_model_14_modified_slim_dynamic2.onnx',
#                     help='Onnx YOLO detection file name')
#     args = parser.parse_args()
    
#     main(args.models_directory, args.onnx_lane_file_name, args.onnx_recognizer_file_name, args.onnx_craft_file_name, args.onnx_yolo_file_name)

from laneLpHandler import LaneLpHandler
from carDetectHandler import CarDetectHandler
import argparse
from lpSrc.build import build_engines

def main(models_directory, onnx_lane_file_name, onnx_recognizer_file_name, onnx_craft_file_name, onnx_yolo_file_name):
    # Build lane detection model
    lanelp_builder = LaneLpHandler()
    lanelp_builder.lane_model_build(models_directory, onnx_lane_file_name)
    
    # Build car detection model
    if onnx_yolo_file_name:
        car_detect_builder = CarDetectHandler()
        car_detect_builder.yolo_model_build(models_directory, onnx_yolo_file_name)
    
    # Build LP recognition models (CRAFT and recognizer)
    _, _ = build_engines(models_directory, onnx_craft_file_name, onnx_recognizer_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_directory', 
                        type=str, 
                        required=True,
                        help='path where onnx is stored and where engine will be saved')
    parser.add_argument('--onnx_lane_file_name', 
                        type=str, 
                        required=True,
                        help='Onnx lane file name')
    parser.add_argument('--onnx_recognizer_file_name', 
                        type=str, 
                        required=True,
                        help='Onnx recognizer file name')
    parser.add_argument('--onnx_craft_file_name', 
                        type=str, 
                        required=True,
                        help='Onnx craft file name')
    parser.add_argument('--onnx_yolo_file_name', 
                    type=str, 
                    default='onnx_holder/fused_removed_960_car_model_14_modified_slim_dynamic2.onnx',
                    help='Onnx YOLO detection file name')
    args = parser.parse_args()
    
    main(args.models_directory, args.onnx_lane_file_name, args.onnx_recognizer_file_name, args.onnx_craft_file_name, args.onnx_yolo_file_name)