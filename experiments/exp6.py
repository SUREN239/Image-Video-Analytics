import torch
import cv2
import numpy as np
from PIL import Image


class ObjectDetector:
    def __init__(self, model_path=None):
        """
        Initialize the object detector
        Args:
            model_path: Path to custom YOLOv5 weights, if None uses pretrained model
        """
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        if model_path:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

        # Set model parameters
        self.model.conf = 0.25  # Confidence threshold
        self.model.iou = 0.45  # NMS IOU threshold
        self.model.classes = None  # Filter by class
        self.model.max_det = 1000  # Maximum detections per image

    def detect_from_image(self, image_path):
        """
        Perform object detection on an image file
        Args:
            image_path: Path to the input image
        Returns:
            processed_img: Image with detection boxes drawn
            detections: List of detection results
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")

        return self._process_image(img)

    def detect_from_webcam(self):
        """
        Perform real-time object detection using webcam
        """
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            processed_frame, detections = self._process_image(frame)

            # Display results
            cv2.imshow('Object Detection', processed_frame)

            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _process_image(self, img):
        """
        Process image through the model and draw detection boxes
        Args:
            img: Input image (numpy array)
        Returns:
            processed_img: Image with detection boxes drawn
            detections: List of detection results
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Make prediction
        results = self.model(img_rgb)

        # Get detections
        detections = results.pandas().xyxy[0].to_dict('records')

        # Draw boxes
        processed_img = img.copy()
        for det in detections:
            x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            conf = det['confidence']
            label = f"{det['name']} {conf:.2f}"

            # Draw rectangle
            cv2.rectangle(processed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label
            cv2.putText(processed_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return processed_img, detections


def main():
    """
    Main function to demonstrate usage
    """
    # Initialize detector
    detector = ObjectDetector()

    # Choose mode
    print("Select mode:")
    print("1. Image file")
    print("2. Webcam")

    mode = input("Enter mode (1 or 2): ")

    if mode == "1":
        # Image file mode
        image_path = input("Enter image path: ")
        try:
            processed_img, detections = detector.detect_from_image(image_path)

            # Display results
            cv2.imshow('Object Detection', processed_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Print detections
            print("\nDetections:")
            for det in detections:
                print(f"Object: {det['name']}, Confidence: {det['confidence']:.2f}")

        except Exception as e:
            print(f"Error: {str(e)}")

    elif mode == "2":
        # Webcam mode
        print("Starting webcam detection (Press 'q' to quit)...")
        detector.detect_from_webcam()

    else:
        print("Invalid mode selected")


if __name__ == "__main__":
    main()