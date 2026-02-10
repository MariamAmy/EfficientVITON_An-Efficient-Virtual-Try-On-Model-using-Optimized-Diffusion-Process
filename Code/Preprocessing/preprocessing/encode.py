from PIL import Image
import base64
from io import BytesIO
import os

def encode_image(image_path):
    """Encodes an image from a file path into a base64 string.

    Args:
        image_path: Path to the image file.

    Returns:
        A base64 encoded string of the image, or None if an error occurs.
    """
    try:
        # Open the image using Pillow
        image = Image.open(image_path)

        # Create a BytesIO buffer to store the image data
        buffered = BytesIO()

        # Save the image to the buffer in a format (e.g., JPEG, PNG)
        image_format = image.format if image.format else "JPEG"
        image.save(buffered, format=image_format)

        # Get the bytes data from the buffer
        image_bytes = buffered.getvalue()

        # Encode the bytes data to base64
        base64_encoded = base64.b64encode(image_bytes).decode("utf-8")

        return base64_encoded
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

if __name__ == '__main__':
    # Example Usage
    image_file = '00000_00.jpg'  # Replace with your actual image path
    # Generate a sample image if the test_image does not exist.
    if not os.path.exists(image_file):
      image = Image.new('RGB', (60, 30), color = 'red')
      image.save(image_file)


    base64_string = encode_image(image_file)

    if base64_string:
        print("Base64 Encoded String:")
        print(base64_string[:100] + "...") # Prints first 100 characters
        # Save to a txt file for reading:
        with open("test.txt","w") as f:
          f.write(base64_string)
    else:
        print("Image encoding failed.")