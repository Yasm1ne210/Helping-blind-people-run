"""This code is meant to be for the standard functioning of our system, through the webcam. ALthough this configuration remains only a modelisation of our vision which involves
smaller, more mobile hardware."""

from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import time
import os
import serial
import glob

class LaneDetector:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def detect_lanes(self, image):
        """
        Detects lane lines in an image
        Returns: original image with lanes drawn and lane points
        """
        h, w, _ = image.shape

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blur, 50, 150)

        # Define region of interest (bottom half of image)
        mask = np.zeros_like(edges)
        roi_vertices = np.array([[(0, h), (0, h/2), (w, h/2), (w, h)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Apply Hough Transform to detect lines
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180,
                                threshold=50, minLineLength=50, maxLineGap=100)

        # Separate lines into left and right lanes
        left_lines = []
        right_lines = []
        lane_points = []

        img_center = w // 2

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate slope
                if x2 - x1 == 0:  # Avoid division by zero
                    continue

                slope = (y2 - y1) / (x2 - x1)

                # Filter by slope - left lanes have negative slope, right lanes positive
                if abs(slope) < 0.3:  # Skip horizontal lines
                    continue

                # Store points for lane guidance
                lane_points.append((x1, y1))
                lane_points.append((x2, y2))

                if slope < 0 and x1 < img_center and x2 < img_center:
                    left_lines.append(line[0])
                elif slope > 0 and x1 > img_center and x2 > img_center:
                    right_lines.append(line[0])

        # Draw lanes on image
        img_with_lanes = image.copy()

        if left_lines:
            for x1, y1, x2, y2 in left_lines:
                cv2.line(img_with_lanes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if right_lines:
            for x1, y1, x2, y2 in right_lines:
                cv2.line(img_with_lanes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Group lane points for output
        lanes = []
        if left_lines:
            left_points = [(x1, y1) for x1, y1, x2, y2 in left_lines] + \
                          [(x2, y2) for x1, y1, x2, y2 in left_lines]
            lanes.append(left_points)

        if right_lines:
            right_points = [(x1, y1) for x1, y1, x2, y2 in right_lines] + \
                           [(x2, y2) for x1, y1, x2, y2 in right_lines]
            lanes.append(right_points)

        return img_with_lanes, lanes

def get_lane_guidance(image, lanes):
    h, w, _ = image.shape
    image_center = w // 2

    # Extract all lane points
    all_points = []
    for lane in lanes:
        all_points.extend(lane)

    if not all_points:
        return "Lane not clear", "UNCLEAR"

    # Check points in the bottom half of the image
    bottom_points = [p for p in all_points if p[1] > h/2]

    if not bottom_points:
        return "Lane not clear", "UNCLEAR"

    # Calculate lane center
    x_coords = [p[0] for p in bottom_points]
    lane_center = sum(x_coords) // len(x_coords)

    # Determine guidance
    if abs(image_center - lane_center) < 30:
        return "You're centered", "CENTER"
    elif image_center > lane_center:
        return "Move left", "LEFT"
    else:
        return "Move right", "RIGHT"

def get_object_guidance(detections):
    """
    Analyze YOLO detections and provide guidance for a self-driving car
    """
    guidance = []
    arduino_code = None

    # Define classes to monitor (0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck)
    monitored_classes = {0: "pedestrian", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    for detection in detections:
        for i, (box, conf, cls) in enumerate(zip(detection.boxes.xyxy, detection.boxes.conf, detection.boxes.cls)):
            cls_id = int(cls.item())

            # Check if detected object is in our monitored list
            if cls_id in monitored_classes:
                obj_type = monitored_classes[cls_id]
                x1, y1, x2, y2 = box.tolist()

                # Calculate box center
                center_x = (x1 + x2) / 2

                # Calculate object distance (approximation based on box height)
                obj_height = y2 - y1
                img_height = detection.orig_shape[0]

                # Rough distance estimation (closer = larger box)
                distance_factor = 1 - (obj_height / img_height)

                # Position relative to frame (left/center/right)
                img_width = detection.orig_shape[1]
                if center_x < img_width / 3:
                    position = "left"
                elif center_x > 2 * img_width / 3:
                    position = "right"
                else:
                    position = "ahead"

                # Create guidance message based on object type, distance and position
                if distance_factor < 0.5:  # Close object
                    if obj_type == "pedestrian":
                        msg = f"CAUTION: {obj_type} {position}, possible stop needed"
                        guidance.append(msg)
                        arduino_code = msg  # Send full message to Arduino
                    else:
                        msg = f"CAUTION: {obj_type} {position}, slow down"
                        guidance.append(msg)
                        arduino_code = msg  # Send full message to Arduino
                elif distance_factor < 0.7:  # Medium distance
                    guidance.append(f"{obj_type} {position}, monitor")

    # Return first guidance if any (prioritize warnings)
    for g in guidance:
        if "CAUTION" in g:
            return g, arduino_code

    return guidance[0] if guidance else "Road clear", None

def find_arduino_port():
    """
    Find the Arduino serial port automatically
    """
    # Common patterns for Arduino ports
    if os.name == 'nt':  # Windows
        ports = ['COM%s' % (i + 1) for i in range(256)]
    else:  # Linux/Mac
        ports = glob.glob('/dev/tty[A-Za-z]*')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass

    # Look for Arduino-specific port names
    for port in result:
        if 'arduino' in port.lower() or 'usbmodem' in port.lower() or 'cu.usb' in port.lower():
            return port

    # If no Arduino-specific port is found, return the first available
    return result[0] if result else None

def process_frame(frame, model, lane_detector, arduino_ser):
    """Process a single frame with YOLO and lane detection and send commands to Arduino"""
    # Detect objects with YOLO
    results = model(frame)

    # Detect lanes
    img_with_lanes, lanes = lane_detector.detect_lanes(frame)

    # Get guidance from both systems
    lane_guide, lane_arduino_cmd = get_lane_guidance(frame, lanes)
    obj_guide, obj_arduino_cmd = get_object_guidance(results)

    # Draw YOLO detections on top of lane detection
    img_with_detections = results[0].plot()

    # Combine the images (lane detection is the base, YOLO detections overlaid)
    alpha = 0.7  # Transparency factor
    combined = cv2.addWeighted(img_with_lanes, alpha, img_with_detections, 1-alpha, 0)

    # Add guidance text
    cv2.putText(combined, f"Lane: {lane_guide}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(combined, f"Objects: {obj_guide}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Prioritize object warnings over lane guidance for Arduino
    arduino_cmd = obj_arduino_cmd if obj_arduino_cmd else lane_arduino_cmd

    # Send command to Arduino if available
    if arduino_ser and arduino_cmd:
        try:
            arduino_ser.write((arduino_cmd + '\n').encode())
            cv2.putText(combined, f"Arduino: {arduino_cmd}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        except Exception as e:
            cv2.putText(combined, f"Arduino error: {str(e)}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return combined

def main():
    # Check if YOLO model exists, if not download it
    model_path = 'yolov8n.pt'
    if not os.path.exists(model_path):
        print("Downloading YOLOv8 model...")
        import urllib.request
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded successfully!")

    # Initialize YOLOv8 model with small size for better speed
    print("Initializing YOLO model...")
    model = YOLO(model_path)
    lane_detector = LaneDetector()
    print("Model initialized successfully!")

    # Initialize Arduino connection
    arduino_port = find_arduino_port()
    arduino_ser = None

    if arduino_port:
        print(f"Arduino found on port {arduino_port}")
        try:
            arduino_ser = serial.Serial(arduino_port, 9600, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            print("Arduino connected successfully!")
        except Exception as e:
            print(f"Error connecting to Arduino: {e}")
            print("Continuing without Arduino support.")
    else:
        print("Arduino not found. Continuing without Arduino support.")
        print("Please check connections and try again.")

    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam, try other indices if not working

    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam. Try changing the camera index.")
        return

    print("Webcam opened successfully!")

    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    start_time = time.time()
    fps = 0

    print("Press 'q' to quit")

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from webcam")
            break

        # Process frame
        frame_count += 1

        # Process every 2nd frame for better performance (adjust as needed)
        if frame_count % 2 == 0:
            result_frame = process_frame(frame, model, lane_detector, arduino_ser)

            # Calculate and display FPS
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                start_time = time.time()

            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Display the result
            cv2.imshow('YOLO + Lane Detection', result_frame)
        else:
            # Just display original frame for skipped frames
            cv2.imshow('YOLO + Lane Detection', frame)

        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    if arduino_ser:
        arduino_ser.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed")

if __name__ == "__main__":
    main()


