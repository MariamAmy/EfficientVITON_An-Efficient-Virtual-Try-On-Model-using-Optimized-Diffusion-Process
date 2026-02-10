import base64
from PIL import Image
from io import BytesIO
import os

def decode_image(base64_string, output_path):
    """Decodes a base64 string into an image file.

    Args:
        base64_string: The base64 encoded string of the image.
        output_path: The path where the decoded image should be saved.

    Returns:
        True if the decoding and saving was successful, False otherwise.
    """
    try:
        # Decode the base64 string to bytes
        image_bytes = base64.b64decode(base64_string)

        # Create a BytesIO object to read the image bytes
        image_buffer = BytesIO(image_bytes)

        # Open the image from the bytes buffer
        image = Image.open(image_buffer)

        # Save the image to the specified output path
        image.save(output_path)
        return True

    except Exception as e:
        print(f"Error decoding image: {e}")
        return False

if __name__ == '__main__':
    # Example Usage
    base64_file = "test.txt"  # Text file containing the base64 string
    output_file = 'decoded_image.jpg'  # Replace with the desired path
    # Read the base64 string from the file
    if os.path.exists(base64_file):
     with open(base64_file,"r") as f:
         base64_string = f.read()
    else:
        print ("There is no file called test.txt")
        exit()



    if decode_image(base64_string, output_file):
        print(f"Image successfully decoded and saved to {output_file}")
    else:
        print("Image decoding failed.")