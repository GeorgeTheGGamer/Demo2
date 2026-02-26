"""
object_detector.py - Lightweight Object Detection using MobileNet-SSD v2 (TFLite)
Optimized for Raspberry Pi 3 and other resource-constrained devices.

Model: MobileNet-SSD v2 quantized (INT8) for fast inference
Framework: TensorFlow Lite Runtime
"""
import math

import cv2
import numpy as np
import os

# Try to import tflite_runtime first (lighter), fall back to full tensorflow
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# COCO dataset class labels (90 classes)
COCO_LABELS = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench',
    15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep',
    20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe',
    26: 'backpack', 27: 'umbrella', 30: 'handbag', 31: 'tie', 32: 'suitcase',
    33: 'frisbee', 34: 'skis', 35: 'snowboard', 36: 'sports ball', 37: 'kite',
    38: 'baseball bat', 39: 'baseball glove', 40: 'skateboard', 41: 'surfboard',
    42: 'tennis racket', 43: 'bottle', 45: 'wine glass', 46: 'cup',
    47: 'fork', 48: 'knife', 49: 'spoon', 50: 'bowl', 51: 'banana',
    52: 'apple', 53: 'sandwich', 54: 'orange', 55: 'broccoli', 56: 'carrot',
    57: 'hot dog', 58: 'pizza', 59: 'donut', 60: 'cake', 61: 'chair',
    62: 'couch', 63: 'potted plant', 64: 'bed', 66: 'dining table',
    69: 'toilet', 71: 'tv', 72: 'laptop', 73: 'mouse', 74: 'remote',
    75: 'keyboard', 76: 'cell phone', 77: 'microwave', 78: 'oven',
    79: 'toaster', 80: 'sink', 81: 'refrigerator', 83: 'book', 84: 'clock',
    85: 'vase', 86: 'scissors', 87: 'teddy bear', 88: 'hair drier', 89: 'toothbrush'
}


def at_right(x, y, left_pts):
    """
    Check if point (x, y) is to the right of the left lane defined by left_pts.
    Args:
        x: x-coordinate of the point to check
        y: y-coordinate of the point to check
        left_pts: points defining the left lane (list of (x, y) tuples)

    Returns:
        True if the point is to the right of the left lane, False otherwise.
    """
    for i in range(len(left_pts) - 1):
        x1, y1 = left_pts[i]
        x2, y2 = left_pts[i + 1]
        if y1 >= y >= y2 or y2 >= y >= y1:
            return x > x1
    return False


def at_left(x, y, right_pts):
    """
    Check if point (x, y) is to the left of the right lane defined by right_pts.
    Args:
        x: x-coordinate of the point to check
        y: y-coordinate of the point to check
        right_pts: points defining the right lane (list of (x, y) tuples)

    Returns:
        True if the point is to the left of the right lane, False otherwise.
    """
    for i in range(len(right_pts) - 1):
        x1, y1 = right_pts[i]
        x2, y2 = right_pts[i + 1]
        if y1 >= y >= y2 or y2 >= y >= y1:
            return x < x1
    return False


class MobileNetSSDDetector:
    """MobileNet-SSD v2 object detector using TensorFlow Lite.
    
    Optimized for Raspberry Pi 3 with INT8 quantization.
    Detects 90 COCO object classes including persons, vehicles, animals.
    """
    
    def __init__(self, model_path='ssd_mobilenet_v2.tflite', num_threads=4, 
                 conf_threshold=0.5, target_classes=None):
        """Initialize the MobileNet-SSD detector.
        
        Args:
            model_path (str): Path to the TFLite model file.
            num_threads (int): Number of CPU threads for inference.
            conf_threshold (float): Minimum confidence for detections.
            target_classes (list): List of class IDs to detect (None = all classes).
                                   Use [0] for person-only detection.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes  # e.g., [0] for person only
        self.labels = COCO_LABELS
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"[WARN] Model not found at {model_path}")
            print("[INFO] Download with: wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip")
            self.interpreter = None
            return
        
        # Initialize TFLite interpreter
        self.interpreter = Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        
        # Get input/output tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape (typically 300x300 or 320x320 for MobileNet-SSD)
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        
        # Check input type (uint8 for quantized, float32 for non-quantized)
        self.is_quantized = self.input_details[0]['dtype'] == np.uint8
        
        print(f"[INFO] MobileNet-SSD loaded: {self.input_width}x{self.input_height}, "
              f"quantized={self.is_quantized}, threads={num_threads}")
    
    def preprocess(self, frame):
        """Preprocess frame for model input.
        
        Args:
            frame (np.ndarray): Input BGR frame.
            
        Returns:
            np.ndarray: Preprocessed input tensor.
        """
        # Resize to model input size
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Add batch dimension
        input_data = np.expand_dims(rgb, axis=0)
        
        # Handle quantized vs float models
        if self.is_quantized:
            input_data = input_data.astype(np.uint8)
        else:
            input_data = (input_data.astype(np.float32) - 127.5) / 127.5
        
        return input_data
    
    def detect(self, frame):
        """Detect objects in frame.
        
        Args:
            frame (np.ndarray): Input BGR frame.
            
        Returns:
            list: List of detections as [x, y, w, h, confidence, class_id, label]
        """
        if self.interpreter is None:
            return []
        
        h, w = frame.shape[:2]
        
        # Preprocess
        input_data = self.preprocess(frame)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get output tensors
        # Output format varies by model, common formats:
        # - boxes: [1, num_detections, 4] normalized [ymin, xmin, ymax, xmax]
        # - classes: [1, num_detections]
        # - scores: [1, num_detections]
        # - num_detections: scalar
        
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        detections = []
        
        for i in range(len(scores)):
            if scores[i] < self.conf_threshold:
                continue
            
            class_id = int(classes[i])
            
            # Filter by target classes if specified
            if self.target_classes is not None and class_id not in self.target_classes:
                continue
            
            # Convert normalized coordinates to pixel coordinates
            ymin, xmin, ymax, xmax = boxes[i]
            
            x = int(xmin * w)
            y = int(ymin * h)
            box_w = int((xmax - xmin) * w)
            box_h = int((ymax - ymin) * h)
            
            # Clamp to frame boundaries
            x = max(0, x)
            y = max(0, y)
            box_w = min(box_w, w - x)
            box_h = min(box_h, h - y)
            
            label = self.labels.get(class_id, f'class_{class_id}')
            
            detections.append([x, y, box_w, box_h, float(scores[i]), class_id, label])
        
        return detections
    
    def detect_persons(self, frame):
        """Convenience method to detect only persons.
        
        Args:
            frame (np.ndarray): Input BGR frame.
            
        Returns:
            list: List of person detections as [x, y, w, h].
        """
        # Temporarily set target to person only
        old_target = self.target_classes
        self.target_classes = [0]  # 0 = person in COCO
        
        detections = self.detect(frame)
        
        self.target_classes = old_target
        
        # Return in simple format [x, y, w, h] for compatibility
        return [[d[0], d[1], d[2], d[3]] for d in detections]
    
    def draw_detections(self, frame, detections, show_label=True, show_conf=True):
        """Draw detection boxes on frame.
        
        Args:
            frame (np.ndarray): Input BGR frame.
            detections (list): List of detections from detect().
            show_label (bool): Show class label.
            show_conf (bool): Show confidence score.
            
        Returns:
            np.ndarray: Frame with drawn detections.
        """
        for det in detections:
            x, y, w, h = det[0], det[1], det[2], det[3]
            conf = det[4] if len(det) > 4 else 1.0
            label = det[6] if len(det) > 6 else 'object'
            
            # Color based on class (person = red, others = green)
            color = (0, 0, 255) if label == 'person' else (0, 255, 0)
            
            # Draw box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            if show_label or show_conf:
                text_parts = []
                if show_label:
                    text_parts.append(label)
                if show_conf:
                    text_parts.append(f'{conf:.2f}')
                text = ' '.join(text_parts)
                
                # Background for text
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, y - text_h - 4), (x + text_w, y), color, -1)
                cv2.putText(frame, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def detect_closest(self, detection, left_pts, right_pts):
        """Detect the closest person and return the y-coordinate of the bottom of the box.

        Args:
            detection: detected boxes [[x, y, w, h, confidence, class_id, label] by the model.
            left_pts (list): List of left lane points.
            right_pts (list): List of right lane points.
        Returns:
            y-coordinate of the middle point of the closest person within the lane,
            or negative infinity if no person in lane.
        """
        # Initialize mid_point to a large value (infinity)
        closest_y = -math.inf
        for det in detection:
            x, y, w, h, _, _, _ = det
            mid_x = x+w/2
            bot_y = y+h

            if (at_right(mid_x, bot_y, left_pts) and
                    at_left(mid_x, bot_y, right_pts) and
                    bot_y > closest_y):
                closest_y = bot_y

        return closest_y


class PersonDetector(MobileNetSSDDetector):
    """Specialized detector for person/pedestrian detection only.
    
    Wrapper around MobileNet-SSD that only detects persons (class 0).
    """
    
    def __init__(self, model_path='ssd_mobilenet_v2.tflite', num_threads=4, conf_threshold=0.5):
        super().__init__(
            model_path=model_path,
            num_threads=num_threads,
            conf_threshold=conf_threshold,
            target_classes=[0]  # Person only
        )
    
    def detect(self, frame):
        """Detect persons in frame.
        
        Args:
            frame (np.ndarray): Input BGR frame.
            
        Returns:
            list: List of detections as [x, y, w, h] for compatibility with existing code.
        """
        detections = super().detect(frame)
        # Return in simple format for backward compatibility
        return [[d[0], d[1], d[2], d[3]] for d in detections]


# Global instance for easy import
_detector = None

def get_detector(model_path='ssd_mobilenet_v2.tflite', num_threads=4):
    """Get or create global detector instance."""
    global _detector
    if _detector is None:
        _detector = PersonDetector(model_path, num_threads)
    return _detector


def detect_persons(frame, model_path='ssd_mobilenet_v2.tflite'):
    """Convenience function for person detection.
    
    Args:
        frame (np.ndarray): Input BGR frame.
        model_path (str): Path to TFLite model.
        
    Returns:
        list: List of person bounding boxes as [x, y, w, h].
    """
    detector = get_detector(model_path)
    return detector.detect(frame)


# --- Test / Demo ---
if __name__ == '__main__':
    import time
    
    print("[TEST] Starting MobileNet-SSD Object Detection Test...")
    print("[INFO] Press 'q' to quit")
    
    # Try to find model
    model_paths = [
        'ssd_mobilenet_v2.tflite',
        'detect.tflite',
        'coco_ssd_mobilenet_v1_1.0_quant/detect.tflite'
    ]
    
    model_path = None
    for p in model_paths:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        print("[ERROR] No model found! Download with:")
        print("  wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip")
        print("  unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip")
        exit(1)
    
    detector = MobileNetSSDDetector(model_path, num_threads=4, conf_threshold=0.4)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    fps_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.time()
        
        # Detect objects
        detections = detector.detect(frame)
        
        # Calculate FPS
        elapsed = time.time() - start
        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)
        
        # Draw detections
        frame = detector.draw_detections(frame, detections)
        
        # Show FPS
        cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'Objects: {len(detections)}', (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('MobileNet-SSD Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
