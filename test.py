
import cv2
import requests
import base64
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Directories
images_path = "/home/alikhan/Desktop/projects/project/images"
output_path = "/home/alikhan/Desktop/projects/project/annotated_images"

# Model API Endpoint
url_model = "http://127.0.0.1:8888/predictions/laneLpService"

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Loop through all images in the folder
for image_name in os.listdir(images_path):
    image_path = os.path.join(images_path, image_name)
    annotated_path = os.path.join(output_path, f"annotated_{image_name}")

    # Ensure it's a valid image file
    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue  # Skip non-image files

    print(f"Processing {image_name}...")

    # Read and encode image as Base64
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    # Prepare request payload
    test_data = {
        "generalFrame": image_base64,
        "direction": "front",
        "unique_id": image_name.split('.')[0],
        "enable_filtering": "disable",
        "lp_Coordinates": [[[1314, 713, 1382, 729]]],
        "dateTime": "11"
    }

    # Send POST request to model
    response = requests.post(url_model, json=test_data)

    # Parse the response
    try:
        response_json = response.json()
        print(f"Response for {image_name}: {response_json}")

        # Load the original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw bounding boxes
        for det in response_json.get("detections", []):
            x_min, y_min, x_max, y_max = det["bbox"]
            confidence = det["confidence"]
            class_name = det["class_name"]

            # Draw rectangle
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Add label
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw lane lines
        for line in response_json.get("lines", []):
            start = tuple(line["start"])
            end = tuple(line["end"])
            line_type = line["type"]

            color = (255, 0, 0) if line_type == "edge" else (0, 0, 255)
            cv2.line(image, start, end, color, 3)

        # Convert back to BGR for saving
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Save the annotated image
        cv2.imwrite(annotated_path, image_bgr)

        print(f"✅ Annotated image saved at: {annotated_path}")

    except json.JSONDecodeError:
        print(f"❌ Error decoding JSON response for {image_name}: {response.text}")

print("✅ All images processed!")
