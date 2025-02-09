import cv2
import numpy as np
from scipy.fft import dct, idct


class ImageWatermarking:
    def __init__(self, alpha=0.1):
        """
        Initialize watermarking with strength factor

        Parameters:
        alpha: float - Watermark strength (0.1 is good balance)
        """
        self.alpha = alpha

    def apply_dct(self, image_block):
        """Apply 2D DCT to image block"""
        return dct(dct(image_block.T, norm='ortho').T, norm='ortho')

    def apply_idct(self, dct_block):
        """Apply 2D inverse DCT to frequency block"""
        return idct(idct(dct_block.T, norm='ortho').T, norm='ortho')

    def embed_watermark(self, image_path, watermark_path, output_path):
        """
        Embed watermark into image using DCT

        Parameters:
        image_path: str - Path to original image
        watermark_path: str - Path to watermark image
        output_path: str - Path to save watermarked image
        """
        # Read images
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

        if original is None or watermark is None:
            raise ValueError("Images not found")

        # Resize watermark to match original image
        watermark = cv2.resize(watermark, (original.shape[1], original.shape[0]))

        # Convert to float
        original_float = np.float32(original)
        watermark_float = np.float32(watermark)

        # Block size for DCT
        block_size = 8
        height, width = original.shape
        watermarked = np.zeros_like(original_float)

        # Process each 8x8 block
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                # Get current block
                block = original_float[i:i + block_size, j:j + block_size]
                watermark_block = watermark_float[i:i + block_size, j:j + block_size]

                # Apply DCT
                dct_block = self.apply_dct(block)

                # Embed watermark
                dct_block += self.alpha * watermark_block

                # Apply inverse DCT
                watermarked[i:i + block_size, j:j + block_size] = self.apply_idct(dct_block)

        # Clip values and convert back to uint8
        watermarked = np.clip(watermarked, 0, 255)
        watermarked = np.uint8(watermarked)

        # Save watermarked image
        cv2.imwrite(output_path, watermarked)
        return watermarked

    def extract_watermark(self, original_path, watermarked_path, output_path):
        """
        Extract watermark from watermarked image

        Parameters:
        original_path: str - Path to original image
        watermarked_path: str - Path to watermarked image
        output_path: str - Path to save extracted watermark
        """
        # Read images
        original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        watermarked = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE)

        if original is None or watermarked is None:
            raise ValueError("Images not found")

        # Convert to float
        original_float = np.float32(original)
        watermarked_float = np.float32(watermarked)

        # Block size for DCT
        block_size = 8
        height, width = original.shape
        extracted = np.zeros_like(original_float)

        # Process each 8x8 block
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                # Get current blocks
                original_block = original_float[i:i + block_size, j:j + block_size]
                watermarked_block = watermarked_float[i:i + block_size, j:j + block_size]

                # Apply DCT
                dct_original = self.apply_dct(original_block)
                dct_watermarked = self.apply_dct(watermarked_block)

                # Extract watermark
                extracted[i:i + block_size, j:j + block_size] = (dct_watermarked - dct_original) / self.alpha

        # Normalize and convert to uint8
        extracted = cv2.normalize(extracted, None, 0, 255, cv2.NORM_MINMAX)
        extracted = np.uint8(extracted)

        # Save extracted watermark
        cv2.imwrite(output_path, extracted)
        return extracted


def main_watermarking():
    watermarker = ImageWatermarking(alpha=0.1)

    try:
        # Embed watermark
        watermarker.embed_watermark(
            "D:/GitHub/Image-Video-Analytics/data-sets/image.jpeg",
            "D:/GitHub/Image-Video-Analytics/data-sets/watermark.png",
            "D:/GitHub/Image-Video-Analytics/experiment-results/watermarked.png"
        )
        print("Watermark embedded successfully")

        # Extract watermark
        watermarker.extract_watermark(
            "D:/GitHub/Image-Video-Analytics/data-sets/image.jpeg",
            "D:/GitHub/Image-Video-Analytics/experiment-results/watermarked.png",
            "D:/GitHub/Image-Video-Analytics/experiment-results/extracted_watermark.png"
        )
        print("Watermark extracted successfully")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main_watermarking()