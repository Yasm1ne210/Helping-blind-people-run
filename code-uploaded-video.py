"""This code is the one meant to help our system deal with uploaded videos, when we conduct tests that do not necessarily involve the camera as a source of video or image input."""

!pip install -q ultralytics opencv-python numpy torch torchvision #for Google Colab

from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import time
from google.colab import files
import matplotlib.pyplot as plt


!wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Create custom lane detection model 

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
        
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
       
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
      
        edges = cv2.Canny(blur, 50, 150)
        
        # Define region of interest (bottom half of image)
        mask = np.zeros_like(edges)
        roi_vertices = np.array([[(0, h), (0, h/2), (w, h/2), (w, h)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        

      
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 
                                threshold=50, minLineLength=50, maxLineGap=100)
        
        left_lines = []
        right_lines = []
        lane_points = []
        
        img_center = w // 2
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate slope
                if x2 - x1 == 0: 
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter by slope - left lanes have negative slope, right lanes positive
                if abs(slope) < 0.3:  # Skip horizontal lines
                    continue
                
                
                lane_points.append((x1, y1))
                lane_points.append((x2, y2))
                    
                if slope < 0 and x1 < img_center and x2 < img_center:
                    left_lines.append(line[0])
                elif slope > 0 and x1 > img_center and x2 > img_center:
                    right_lines.append(line[0])
        
       
        img_with_lanes = image.copy()
        
        if left_lines:
            for x1, y1, x2, y2 in left_lines:
                cv2.line(img_with_lanes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        if right_lines:
            for x1, y1, x2, y2 in right_lines:
                cv2.line(img_with_lanes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
       
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


  
model = YOLO('yolov8n.pt')
lane_detector = LaneDetector()

  
uploaded = files.upload()
uploaded_path = list(uploaded.keys())[0]
print("Uploaded:", uploaded_path)
file_ext = os.path.splitext(uploaded_path)[-1]

# Defining processing functions
def get_lane_guidance(image, lanes):
    h, w, _ = image.shape
    image_center = w // 2
    
   
    all_points = []
    for lane in lanes:
        all_points.extend(lane)
    
    if not all_points:
        return "Lane not clear"
    
   
    bottom_points = [p for p in all_points if p[1] > h/2]
    
    if not bottom_points:
        return "Lane not clear"
    
   
    x_coords = [p[0] for p in bottom_points]
    lane_center = sum(x_coords) // len(x_coords)
    

    if abs(image_center - lane_center) < 30:
        return "You're centered"
    elif image_center > lane_center:
        return "Move left"
    else:
        return "Move right"

def get_object_guidance(detections):
    """
    Analyze YOLO detections and provide guidance for a self-driving car
    """
    guidance = []
    
   
    monitored_classes = {0: "pedestrian", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    
    for detection in detections:
        for i, (box, conf, cls) in enumerate(zip(detection.boxes.xyxy, detection.boxes.conf, detection.boxes.cls)):
            cls_id = int(cls.item())
            
            
            if cls_id in monitored_classes:
                obj_type = monitored_classes[cls_id]
                x1, y1, x2, y2 = box.tolist()
                
                center_x = (x1 + x2) / 2
    
                obj_height = y2 - y1
                img_height = detection.orig_shape[0]
                
                distance_factor = 1 - (obj_height / img_height)
                
               
                img_width = detection.orig_shape[1]
                if center_x < img_width / 3:
                    position = "left"
                elif center_x > 2 * img_width / 3:
                    position = "right"
                else:
                    position = "ahead"
                
    
                if distance_factor < 0.5:
                    if obj_type == "pedestrian":
                        guidance.append(f"CAUTION: {obj_type} {position}, possible stop needed")
                    else:
                        guidance.append(f"CAUTION: {obj_type} {position}, slow down")
                elif distance_factor < 0.7:  # Medium distance
     
    for g in guidance:
        if "CAUTION" in g:
            return g
    
    return guidance[0] if guidance else "Road clear"

def process_frame(frame):
    """Process a single frame with YOLO and lane detection """
    
   
    results = model(frame)
    
   
    img_with_lanes, lanes = lane_detector.detect_lanes(frame)
    
   
    lane_guide = get_lane_guidance(frame, lanes)
    obj_guide = get_object_guidance(results)
    
   
    img_with_detections = results[0].plot()
    
  
    alpha = 0.7  # Transparency factor
    combined = cv2.addWeighted(img_with_lanes, alpha, img_with_detections, 1-alpha, 0)
    

    cv2.putText(combined, f"Lane: {lane_guide}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(combined, f"Objects: {obj_guide}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    return combined


if file_ext.lower() in ['.jpg', '.jpeg', '.png']:
    # Process image
    img = cv2.imread(uploaded_path)
    result = process_frame(img)
    

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    

    cv2.imwrite('result.jpg', result)
    files.download('result.jpg')

elif file_ext.lower() in ['.mp4', '.avi', '.mov']:
  
    cap = cv2.VideoCapture(uploaded_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('result_video.mp4', fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
 
        if frame_count % 2 == 0:
            result_frame = process_frame(frame)
            out.write(result_frame)
     
            if frame_count % 20 == 0:
                elapsed = time.time() - start_time
                fps_processing = (frame_count + 1) / elapsed if elapsed > 0 else 0
                percent_done = (frame_count + 1) / total_frames * 100 if total_frames > 0 else 0
                print(f"Processing: {percent_done:.1f}% complete ({frame_count+1}/{total_frames}) - {fps_processing:.1f} FPS")
        else:
         
            out.write(frame)
            
        frame_count += 1
    
    # Clean up
    cap.release()
    out.release()
    print(f"Video processing complete. Output saved as 'result_video.mp4'")
    

    from IPython.display import HTML
    from base64 import b64encode
    
    mp4 = open('result_video.mp4', 'rb').read()
    data_url = f"data:video/mp4;base64,{b64encode(mp4).decode()}"
    display(HTML(f"""
    <video width="640" height="480" controls>
      <source src="{data_url}" type="video/mp4">
    </video>
    """))
    
 
    files.download('result_video.mp4')

else:
    print("Unsupported file type. Please upload .jpg, .jpeg, .png, .mp4, .avi, or .mov file.")
