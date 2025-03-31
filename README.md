### AI-RoadTrack

This project is designed for automated vehicle detection, lane detection, and license plate recognition using deep learning. It includes model serving with TorchServe and provides scripts for preprocessing, model inference, and result visualization.

The system is built using Python and integrates deep learning frameworks such as PyTorch and TensorRT. The service is containerized using Docker for deployment efficiency. Utility scripts are included for handling model configurations, compiling necessary resources, and optimizing inference performance.

To set up the project, install the required dependencies listed in the project files and follow the provided scripts for model handling. The repository does not include precompiled ONNX or TensorRT engine files. Users must generate these files themselves.

This project is intended for research purposes, and users are responsible for compliance with applicable regulations regarding vehicle recognition and data privacy.

Steps
Step 0. Upload your onnx files to service_code/lane_lp_handler/onnx_holder

Step 1. Execute service_compile.sh inside service_code folder (Change the paths)

Step 2. Create Docker image using Dockerfile 

Step 3. Inside torchserve-control.sh replace image name with your docker image name:

IMAGE_NAME="change to your image name"

Now you can create container using "./torchserve-control.sh start" command 

"./torchserve-control.sh test" uses script car_detection_test.py 


