import cv2
import numpy as np
from os import path as osp
import os
import time
import argparse
from tqdm import tqdm

# Record the start time
start_time = time.time()

def get_cloth_mask(image_path):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(image)

    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

    return mask

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = os.listdir(input_folder)
    image_files = [f for f in image_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for image_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        cloth_mask = get_cloth_mask(input_path)

        cv2.imwrite(output_path, cloth_mask)

        print(f"Cloth mask saved at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cloth masks from images")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset folder containing images")
    args = parser.parse_args()

    input_folder = osp.join(args.data_path, 'cloth')
    output_folder = osp.join(args.data_path, 'cloth-mask')
    os.makedirs(output_folder, exist_ok=True)
    
    process_images(input_folder, output_folder)

    # Record the end time
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    # Print the execution time in seconds
    print("Execution time: {:.4f} seconds".format(execution_time))