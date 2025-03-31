import os
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

def build_engine(onnx_model_path, trt_engine_path, model_type, max_workspace_size=1 << 30):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_model_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX model.')
            for error in range(parser.num_errors):
                print(f"{error}: {parser.get_error(error).desc()}")
            exit(1)
    
    profile1 = builder.create_optimization_profile()

    if model_type == "craft":
        # optimization profile for craft.onnx
        profile1.set_shape('x', min=[1, 3, 64, 320], opt=[1, 3, 192, 320], max=[1, 3, 192, 320])
    elif model_type == "recognizer":
        # optimization profile for recognizer.onnx
        profile1.set_shape('input.1', min=[1, 3, 32, 128], opt=[1, 3, 32, 128], max=[1, 3, 32, 128])
    else:
        raise ValueError(f"Unsupported model: {model_type}")
    
    config = builder.create_builder_config()
    config.add_optimization_profile(profile1)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_serialized_network(network, config)

    with open(trt_engine_path, "wb") as f:
        f.write(engine)
    

def build_engines(models_directory, craft_onnx_path, recognizer_onnx_path):
    craft_engine_path = os.path.join(models_directory, "engine_holder", os.path.basename(craft_onnx_path).split('.')[0] + ".engine")
    recognizer_engine_path = os.path.join(models_directory, "engine_holder", os.path.basename(recognizer_onnx_path).split('.')[0] + ".engine")

    if not os.path.exists(craft_engine_path):
        print("Building craft trt engine")
        build_engine(os.path.join(models_directory, craft_onnx_path), craft_engine_path, "craft")
    else:
        print("Craft engine already exists")

    if not os.path.exists(recognizer_engine_path):
        print("Building recognizer trt engine")
        build_engine(os.path.join(models_directory, recognizer_onnx_path), recognizer_engine_path, "recognizer")
    else:
        print("Recognizer engine already exists")

    return craft_engine_path, recognizer_engine_path

