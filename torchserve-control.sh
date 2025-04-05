#!/bin/bash

show_usage() {
    echo "Usage: $0 [start|stop|status|logs]"
    echo "  start  - Start the TorchServe container"
    echo "  stop   - Stop and remove the TorchServe container"
    echo "  status - Check if the TorchServe container is running"
    echo "  logs   - Show logs from the container"
    echo "  test   - Run the car detection test script"
}

if [ $# -lt 1 ]; then
    show_usage
    exit 1
fi

CONTAINER_NAME="service"
IMAGE_NAME="ves-patrol:5.0"

case "$1" in
    start)
        echo "Starting TorchServe container..."
        
        # Checking if container with the same name exists
        if sudo docker ps -a | grep -q $CONTAINER_NAME; then
            echo "Container '$CONTAINER_NAME' already exists. Stopping and removing it..."
            sudo docker stop $CONTAINER_NAME
            sudo docker rm $CONTAINER_NAME
        fi
        
        # Starting
        sudo docker run -d \
          --name $CONTAINER_NAME \
          --gpus all \
          -p 8080:8080 \
          -p 8081:8081 \
          -p 8082:8082 \
          -p 8888:8888 \
          -v "$(pwd)/service_holder:/home/model-server/model-store" \
          -v "$(pwd)/service_kickoff/config.properties:/home/model-server/config.properties" \
          --env TEMP=/tmp \
          $IMAGE_NAME
        
        echo "Container started. Waiting for services to initialize..."
        sleep 10
        
        # Check if container is running
        if sudo docker ps | grep -q $CONTAINER_NAME; then
            echo "Container is running."
            echo "You can now run detection tests with: $0 test"
        else
            echo "ERROR: Container failed to start."
            echo "Check logs with: $0 logs"
        fi
        ;;
        
    stop)
        echo "Stopping TorchServe container..."
        if sudo docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
            sudo docker stop $CONTAINER_NAME
            sudo docker rm $CONTAINER_NAME
            echo "Container stopped and removed."
        else
            echo "Container is not running."
        fi
        ;;
        
    status)
        echo "Checking TorchServe container status..."
        if sudo docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
            echo "Container is RUNNING."
            sudo docker ps -f name=$CONTAINER_NAME
            
            echo -e "\nTesting inference endpoint..."
            curl -s http://localhost:8888/ping || echo "Inference endpoint not responding"
        else
            echo "Container is NOT running."
            if sudo docker ps -a -q -f name=$CONTAINER_NAME | grep -q .; then
                echo "Container exists but is stopped. Start it with: $0 start"
            else
                echo "Container does not exist. Create it with: $0 start"
            fi
        fi
        ;;
        
    logs)
        echo "Showing logs from TorchServe container..."
        sudo docker logs $CONTAINER_NAME
        ;;
        
    test)
        echo "Running car detection test..."
        python3 car_detection_test.py
        ;;
        
    *)
        show_usage
        exit 1
        ;;
esac
