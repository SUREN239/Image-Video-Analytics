import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


class VisualInspectionSystem:
    def __init__(self):
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()

        # Initialize feature matcher
        self.matcher = cv2.BFMatcher()

        # Initialize template database
        self.template_database = {}

    def extract_features(self, image):
        """Extract SIFT features from image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def extract_shape_features(self, contour):
        """Extract shape-based features from contour"""
        # Calculate basic shape features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Calculate shape features
        features = {
            'area': area,
            'perimeter': perimeter,
            'circularity': (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0,
            'convex_hull_area': cv2.contourArea(cv2.convexHull(contour)),
            'solidity': area / cv2.contourArea(cv2.convexHull(contour)) if cv2.contourArea(
                cv2.convexHull(contour)) > 0 else 0
        }

        return features

    def add_template(self, name, image):
        """Add template to database"""
        keypoints, descriptors = self.extract_features(image)
        self.template_database[name] = {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'image': image
        }

    def detect_objects(self, image, threshold=0.7):
        """Detect objects in image by matching with templates"""
        results = []
        image_keypoints, image_descriptors = self.extract_features(image)

        if image_descriptors is None:
            return results

        for template_name, template_data in self.template_database.items():
            if template_data['descriptors'] is None:
                continue

            # Match features
            matches = self.matcher.knnMatch(template_data['descriptors'],
                                            image_descriptors, k=2)

            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < threshold * n.distance:
                    good_matches.append(m)

            if len(good_matches) >= 4:
                # Get matched keypoints
                template_pts = np.float32([template_data['keypoints'][m.queryIdx].pt
                                           for m in good_matches]).reshape(-1, 1, 2)
                image_pts = np.float32([image_keypoints[m.trainIdx].pt
                                        for m in good_matches]).reshape(-1, 1, 2)

                # Find homography
                H, mask = cv2.findHomography(template_pts, image_pts, cv2.RANSAC, 5.0)

                if H is not None:
                    # Get template corners
                    h, w = template_data['image'].shape[:2]
                    template_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                    # Transform corners
                    transformed_corners = cv2.perspectiveTransform(template_corners, H)

                    results.append({
                        'name': template_name,
                        'corners': transformed_corners,
                        'confidence': len(good_matches) / len(matches),
                        'homography': H
                    })

        return results

    def inspect_quality(self, image, object_mask):
        """Perform quality inspection on detected object"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Find contours
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        main_contour = max(contours, key=cv2.contourArea)

        # Extract shape features
        shape_features = self.extract_shape_features(main_contour)

        # Calculate texture features
        object_roi = cv2.bitwise_and(gray, gray, mask=object_mask)
        texture_features = {
            'mean_intensity': np.mean(object_roi[object_mask > 0]),
            'std_intensity': np.std(object_roi[object_mask > 0]),
            'uniformity': np.sum(np.square(np.histogram(object_roi[object_mask > 0],
                                                        bins=256)[0] / np.sum(object_mask)))
        }

        # Combine features
        quality_metrics = {**shape_features, **texture_features}

        return quality_metrics

    def visualize_results(self, image, detections):
        """Visualize detection results"""
        output = image.copy()

        for detection in detections:
            # Draw bounding box
            corners = detection['corners']
            corners = np.int32(corners)
            cv2.polylines(output, [corners], True, (0, 255, 0), 2)

            # Add label
            label = f"{detection['name']} ({detection['confidence']:.2f})"
            cv2.putText(output, label, tuple(corners[0][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return output


def create_sample_images():
    """Create sample template and test images with clear visual differences"""
    # Create template image (a blue rectangle with a red circle inside)
    template = np.ones((300, 300, 3), dtype=np.uint8) * 255  # White background
    # Draw blue rectangle
    cv2.rectangle(template, (50, 50), (250, 250), (255, 0, 0), -1)
    # Draw red circle
    cv2.circle(template, (150, 150), 50, (0, 0, 255), -1)
    cv2.imwrite('../experiment-results/template.jpg', template)

    # Create test image with multiple objects including the template pattern
    test_image = np.ones((600, 800, 3), dtype=np.uint8) * 255  # White background

    # Draw the template pattern at different locations
    # Good pattern (similar to template)
    cv2.rectangle(test_image, (50, 50), (250, 250), (255, 0, 0), -1)
    cv2.circle(test_image, (150, 150), 50, (0, 0, 255), -1)

    # Defective pattern (different color)
    cv2.rectangle(test_image, (300, 50), (500, 250), (0, 255, 0), -1)  # Green rectangle
    cv2.circle(test_image, (400, 150), 50, (0, 0, 255), -1)

    # Different pattern (missing circle)
    cv2.rectangle(test_image, (550, 50), (750, 250), (255, 0, 0), -1)

    # Add some noise to make it more realistic
    noise = np.random.normal(0, 5, test_image.shape).astype(np.uint8)
    test_image = cv2.add(test_image, noise)

    cv2.imwrite('../experiment-results/test_image.jpg', test_image)

    return template, test_image


def display_results(template, test_image, result_image):
    """Display the template, test image, and results side by side"""
    plt.figure(figsize=(15, 5))

    # Display template
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    plt.title('Template (Reference Pattern)')
    plt.axis('off')

    # Display test image
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title('Test Image (Multiple Patterns)')
    plt.axis('off')

    # Display result
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Detection Results')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # Create sample images
    print("Creating sample images...")
    template, test_image = create_sample_images()

    # Initialize the inspection system
    print("Initializing visual inspection system...")
    inspector = VisualInspectionSystem()

    # Add template
    print("Adding template to database...")
    inspector.add_template('reference_pattern', template)

    # Detect objects
    print("Detecting objects...")
    detections = inspector.detect_objects(test_image, threshold=0.7)

    # Visualize results
    print("Visualizing results...")
    result_image = inspector.visualize_results(test_image, detections)

    # Display all images
    display_results(template, test_image, result_image)

    # Print detection results
    print("\nDetection Results:")
    for i, detection in enumerate(detections):
        print(f"\nObject {i + 1}:")
        print(f"Confidence: {detection['confidence']:.2f}")

        # Create mask for the detected object
        mask = np.zeros(test_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(detection['corners'])], 255)

        # Inspect quality
        quality_metrics = inspector.inspect_quality(test_image, mask)
        print("Quality Metrics:")
        for metric, value in quality_metrics.items():
            print(f"{metric}: {value:.2f}")


if __name__ == "__main__":
    main()