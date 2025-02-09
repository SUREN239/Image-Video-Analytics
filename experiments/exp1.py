import cv2
import numpy as np
import matplotlib.pyplot as plt


def build_t_pyramid(image, levels=4):
    """
    Build a T-pyramid (Gaussian pyramid) from an input image.

    Parameters:
    image: numpy array - Input image
    levels: int - Number of pyramid levels to generate

    Returns:
    list of numpy arrays - The pyramid levels from original to smallest
    """
    # Convert image to float32 for better precision
    current = image.astype(np.float32)
    pyramid = [current]

    # Generate pyramid levels
    for i in range(levels - 1):
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(current, (5, 5), 0)
        # Downsample by taking every second pixel
        current = blurred[::2, ::2]
        pyramid.append(current)

    return pyramid


def display_pyramid(pyramid):
    """
    Display all levels of the pyramid side by side.

    Parameters:
    pyramid: list of numpy arrays - The pyramid levels to display
    """
    fig, axes = plt.subplots(1, len(pyramid), figsize=(15, 3))

    for i, level in enumerate(pyramid):
        if len(level.shape) == 2:  # Grayscale image
            axes[i].imshow(level, cmap='gray')
        else:  # Color image
            axes[i].imshow(cv2.cvtColor(level.astype(np.uint8), cv2.COLOR_BGR2RGB))
        axes[i].axis('off')
        axes[i].set_title(f'Level {i}')

    plt.tight_layout()
    plt.show()


def main():
    # Read an image
    image = cv2.imread('D:/GitHub/Image-Video-Analytics/data-sets/image.jpeg')
    if image is None:
        print("Error: Could not read image")
        return

    # Build pyramid
    pyramid = build_t_pyramid(image)

    # Display results
    display_pyramid(pyramid)

    # Optionally save pyramid levels
    for i, level in enumerate(pyramid):
        cv2.imwrite(f'pyramid_level_{i}.jpg', level)


if __name__ == "__main__":
    main()