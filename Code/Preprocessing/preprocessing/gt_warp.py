from PIL import Image
import os
from os import path as osp
import argparse

def extract_color_to_mask(image_dir, palette_index, output_dir):
    """
    Extracts a specific color from a P mode image and saves it as a binary mask.

    Parameters:
        image_dir (str): Path to the directory containing input images (P mode).
        palette_index (int): The palette index of the color to extract.
        output_dir (str): Path to the directory to save the binary masks (L mode, JPG format).
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):  # Adjust if other image formats are needed
            img_path = os.path.join(image_dir, filename)

            # Open the image in P mode
            img = Image.open(img_path)

            if img.mode != 'P':
                print(f"Skipping {filename} (not in P mode)")
                continue

            # Create a new image of the same size in L mode (binary mask)
            mask = Image.new('L', img.size)

            # Load the pixel data
            img_pixels = img.load()
            mask_pixels = mask.load()

            # Iterate over all pixels
            for y in range(img.size[1]):
                for x in range(img.size[0]):
                    # If the pixel matches the target palette index, set the mask to white (255)
                    mask_pixels[x, y] = 255 if img_pixels[x, y] in palette_index else 0

            # Change the file extension to .jpg
            output_filename = filename.replace(".png", ".jpg")
            output_path = os.path.join(output_dir, output_filename)

            # Save the mask as a JPG
            mask.save(output_path, format="JPEG")
            print(f"Saved mask for {filename} as {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cloth masks from images")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset folder containing images")
    args = parser.parse_args()

    input_folder = osp.join(args.data_path, 'image-parse-v3')
    output_folder = osp.join(args.data_path, 'gt_cloth_warped_mask')
    os.makedirs(output_folder, exist_ok=True)
    extract_color_to_mask(input_folder, (5, 6, 7), output_folder)