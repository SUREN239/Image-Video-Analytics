import numpy as np
import cv2
import matplotlib.pyplot as plt


class QuadTreeNode:
    def __init__(self, x, y, width, height, depth=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.depth = depth
        self.children = []
        self.is_leaf = True
        self.mean_intensity = None


def build_quadtree(image, x, y, width, height, threshold=5, max_depth=7, depth=0):
    """
    Recursively build a quadtree for the given image region based on intensity homogeneity.

    Parameters:
    image: numpy array - Input grayscale image
    x, y: int - Top-left corner coordinates of the current region
    width, height: int - Dimensions of the current region
    threshold: float - Maximum allowed intensity variance for a homogeneous region
    max_depth: int - Maximum allowed depth of the quadtree
    depth: int - Current depth in the recursion

    Returns:
    QuadTreeNode - Root node of the quadtree
    """
    node = QuadTreeNode(x, y, width, height, depth)

    # Get the region of interest
    region = image[y:y + height, x:x + width]

    # Calculate mean intensity and variance
    mean_intensity = np.mean(region)
    variance = np.var(region)

    node.mean_intensity = mean_intensity

    # Check if we should split this node
    if variance > threshold and depth < max_depth and width > 1 and height > 1:
        # Calculate new dimensions
        new_width = width // 2
        new_height = height // 2

        # Create four children (quadrants)
        node.is_leaf = False
        node.children = [
            build_quadtree(image, x, y, new_width, new_height,
                           threshold, max_depth, depth + 1),  # Top-left
            build_quadtree(image, x + new_width, y, new_width, new_height,
                           threshold, max_depth, depth + 1),  # Top-right
            build_quadtree(image, x, y + new_height, new_width, new_height,
                           threshold, max_depth, depth + 1),  # Bottom-left
            build_quadtree(image, x + new_width, y + new_height, new_width, new_height,
                           threshold, max_depth, depth + 1)  # Bottom-right
        ]

    return node


def visualize_quadtree(image, root, output_image=None):
    """
    Visualize the quadtree decomposition by drawing boundaries.

    Parameters:
    image: numpy array - Original image
    root: QuadTreeNode - Root node of the quadtree
    output_image: numpy array - Image to draw on (created if None)

    Returns:
    numpy array - Image with quadtree boundaries drawn
    """
    if output_image is None:
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def draw_node(node):
        if not node.is_leaf:
            # Draw rectangle for non-leaf nodes
            cv2.rectangle(output_image,
                          (node.x, node.y),
                          (node.x + node.width, node.y + node.height),
                          (0, 255, 0), 1)
            # Recursively draw children
            for child in node.children:
                draw_node(child)

    draw_node(root)
    return output_image


def reconstruct_image(root, original_shape):
    """
    Reconstruct the image from the quadtree representation.

    Parameters:
    root: QuadTreeNode - Root node of the quadtree
    original_shape: tuple - Shape of the original image

    Returns:
    numpy array - Reconstructed image
    """
    reconstructed = np.zeros(original_shape, dtype=np.uint8)

    def fill_region(node):
        if node.is_leaf:
            reconstructed[node.y:node.y + node.height,
            node.x:node.x + node.width] = node.mean_intensity
        else:
            for child in node.children:
                fill_region(child)

    fill_region(root)
    return reconstructed


def main():
    # Read and convert image to grayscale
    image = cv2.imread('D:/GitHub/Image-Video-Analytics/data-sets/image.jpeg', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not read image")
        return

    # Build quadtree
    root = build_quadtree(image, 0, 0, image.shape[1], image.shape[0], threshold=10)

    # Visualize the decomposition
    visualization = visualize_quadtree(image, root)

    # Reconstruct image from quadtree
    reconstructed = reconstruct_image(root, image.shape)

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
    plt.title('Quadtree Decomposition')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(reconstructed, cmap='gray')
    plt.title('Reconstructed Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()