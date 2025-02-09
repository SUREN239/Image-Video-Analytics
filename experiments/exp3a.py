import cv2
import numpy as np


class ImageSteganography:
    def __init__(self):
        self.delimiter = "####"  # Message delimiter

    def to_binary(self, data):
        """Convert data to binary format"""
        if isinstance(data, str):
            return ''.join([format(ord(i), "08b") for i in data])
        elif isinstance(data, bytes):
            return ''.join([format(i, "08b") for i in data])
        elif isinstance(data, np.ndarray):
            return [format(i, "08b") for i in data]

    def encode(self, image_path, secret_data, output_path):
        """
        Encode secret data into image using LSB steganography

        Parameters:
        image_path: str - Path to cover image
        secret_data: str - Secret message to hide
        output_path: str - Path to save encoded image
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found")

        # Maximum bytes to encode
        n_bytes = image.shape[0] * image.shape[1] * 3 // 8
        if len(secret_data) > n_bytes:
            raise ValueError("Error: Insufficient bytes, need bigger image or less data")

        secret_data += self.delimiter  # Add delimiter
        binary_secret_data = self.to_binary(secret_data)
        data_index = 0

        # Flatten the image
        data_len = len(binary_secret_data)

        for row in image:
            for pixel in row:
                for color in range(3):  # RGB channels
                    if data_index < data_len:
                        # Modify LSB of each color channel
                        pixel[color] = int(bin(pixel[color])[:-1] + binary_secret_data[data_index], 2)
                        data_index += 1

        cv2.imwrite(output_path, image)
        return True

    def decode(self, image_path):
        """
        Decode secret data from steganographic image

        Parameters:
        image_path: str - Path to steganographic image

        Returns:
        str - Decoded secret message
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found")

        binary_data = ""
        for row in image:
            for pixel in row:
                for color in pixel:
                    # Extract LSB from each color channel
                    binary_data += bin(color)[-1]

        # Convert binary to ASCII
        all_bytes = [binary_data[i: i + 8] for i in range(0, len(binary_data), 8)]
        decoded_data = ""

        for byte in all_bytes:
            decoded_data += chr(int(byte, 2))
            if decoded_data[-len(self.delimiter):] == self.delimiter:
                return decoded_data[:-len(self.delimiter)]

        return decoded_data


def main_steganography():
    stego = ImageSteganography()

    # Example usage
    try:
        # Encoding
        secret_message = "This is a secret message from Suren!"
        stego.encode("D:/GitHub/Image-Video-Analytics/data-sets/image.jpeg", secret_message, "D:/GitHub/Image-Video-Analytics/experiment-results/encoded.png")
        print("Encoding completed")

        # Decoding
        decoded_message = stego.decode("D:/GitHub/Image-Video-Analytics/experiment-results/encoded.png")
        print(f"Decoded message: {decoded_message}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main_steganography()