import cv2
import numpy as np
import matplotlib.pyplot as plt


class GeometricTransforms:
    @staticmethod
    def rotate_image(image, angle_degrees, center=None):
        """
        Rotate image around a center point

        Parameters:
        image: numpy array - Input image
        angle_degrees: float - Rotation angle in degrees
        center: tuple - Center point (x,y) for rotation

        Returns:
        numpy array - Rotated image
        """
        if center is None:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

        # Calculate new image dimensions
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        height, width = image.shape[:2]
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        # Adjust rotation matrix
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Perform rotation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
        return rotated_image

    @staticmethod
    def scale_image(image, scale_x, scale_y):
        """
        Scale image in x and y directions

        Parameters:
        image: numpy array - Input image
        scale_x: float - Scale factor in x direction
        scale_y: float - Scale factor in y direction

        Returns:
        numpy array - Scaled image
        """
        height, width = image.shape[:2]
        new_width = int(width * scale_x)
        new_height = int(height * scale_y)
        scaled_image = cv2.resize(image, (new_width, new_height))
        return scaled_image

    @staticmethod
    def skew_image(image, skew_x, skew_y):
        """
        Apply skewing transformation

        Parameters:
        image: numpy array - Input image
        skew_x: float - Skew factor in x direction
        skew_y: float - Skew factor in y direction

        Returns:
        numpy array - Skewed image
        """
        height, width = image.shape[:2]

        # Create skew matrix
        skew_matrix = np.float32([
            [1, skew_x, 0],
            [skew_y, 1, 0]
        ])

        # Calculate new dimensions
        new_width = int(width + height * abs(skew_x))
        new_height = int(height + width * abs(skew_y))

        # Apply skewing
        skewed_image = cv2.warpAffine(image, skew_matrix, (new_width, new_height))
        return skewed_image

    @staticmethod
    def affine_transform(image, src_points, dst_points):
        """
        Apply affine transform from three point correspondences

        Parameters:
        image: numpy array - Input image
        src_points: numpy array - Three source points
        dst_points: numpy array - Three destination points

        Returns:
        numpy array - Transformed image
        """
        # Get affine transform matrix
        affine_matrix = cv2.getAffineTransform(src_points, dst_points)

        # Calculate new dimensions
        height, width = image.shape[:2]
        corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
        transformed_corners = cv2.transform(corners, affine_matrix)

        min_x = np.min(transformed_corners[:, 0, 0])
        max_x = np.max(transformed_corners[:, 0, 0])
        min_y = np.min(transformed_corners[:, 0, 1])
        max_y = np.max(transformed_corners[:, 0, 1])

        new_width = int(max_x - min_x)
        new_height = int(max_y - min_y)

        # Adjust transformation matrix for new dimensions
        affine_matrix[0, 2] -= min_x
        affine_matrix[1, 2] -= min_y

        # Apply transformation
        transformed_image = cv2.warpAffine(image, affine_matrix, (new_width, new_height))
        return transformed_image

    @staticmethod
    def bilinear_transform(image, src_points, dst_points):
        """
        Apply bilinear transform from four point correspondences

        Parameters:
        image: numpy array - Input image
        src_points: numpy array - Four source points
        dst_points: numpy array - Four destination points

        Returns:
        numpy array - Transformed image
        """

        def get_bilinear_matrix(src, dst):
            # Create matrices for solving coefficients
            A = np.zeros((8, 8))
            b = np.zeros(8)

            for i in range(4):
                x, y = src[i]
                u, v = dst[i]
                A[i] = [x, y, 1, 0, 0, 0, -x * u, -y * u]
                A[i + 4] = [0, 0, 0, x, y, 1, -x * v, -y * v]
                b[i] = u
                b[i + 4] = v

            # Solve for coefficients
            coeffs = np.linalg.solve(A, b)
            return coeffs

        # Get transformation coefficients
        coeffs = get_bilinear_matrix(src_points, dst_points)

        # Calculate new dimensions
        height, width = image.shape[:2]
        corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        # Create meshgrid for transformation
        y, x = np.mgrid[0:height, 0:width]
        x = x.flatten()
        y = y.flatten()

        # Apply bilinear transformation
        denom = coeffs[6] * x + coeffs[7] * y + 1
        new_x = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / denom
        new_y = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / denom

        # Reshape coordinates
        map_x = new_x.reshape(height, width).astype(np.float32)
        map_y = new_y.reshape(height, width).astype(np.float32)

        # Apply transformation using remap
        transformed_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        return transformed_image


def create_sample_image():
    """Create a sample image with a simple pattern"""
    image = np.ones((300, 300, 3), dtype=np.uint8) * 255
    cv2.rectangle(image, (100, 100), (200, 200), (0, 0, 255), -1)
    cv2.circle(image, (150, 150), 30, (255, 0, 0), -1)
    return image


def display_results(original, transformed, title):
    """Display original and transformed images side by side"""
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
    plt.title(f'After {title}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # Create instance of transforms class
    transforms = GeometricTransforms()

    # Create sample image
    image = create_sample_image()

    # 1. Rotation
    rotated = transforms.rotate_image(image, 45)
    display_results(image, rotated, "Rotation (45 degrees)")

    # 2. Scaling
    scaled = transforms.scale_image(image, 1.5, 0.75)  # Scale x by 1.5, y by 0.75
    display_results(image, scaled, "Scaling (x:1.5, y:0.75)")

    # 3. Skewing
    skewed = transforms.skew_image(image, 0.3, 0.2)
    display_results(image, skewed, "Skewing (x:0.3, y:0.2)")

    # 4. Affine Transform
    height, width = image.shape[:2]
    src_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])
    dst_points = np.float32([[width * 0.2, height * 0.1], [width * 0.9, height * 0.2], [width * 0.1, height * 0.9]])
    affine = transforms.affine_transform(image, src_points, dst_points)
    display_results(image, affine, "Affine Transform")

    # 5. Bilinear Transform
    src_points = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    dst_points = np.float32([[width * 0.1, height * 0.1], [width * 0.9, height * 0.1],
                             [width * 0.9, height * 0.9], [width * 0.1, height * 0.9]])
    bilinear = transforms.bilinear_transform(image, src_points, dst_points)
    display_results(image, bilinear, "Bilinear Transform")


if __name__ == "__main__":
    main()