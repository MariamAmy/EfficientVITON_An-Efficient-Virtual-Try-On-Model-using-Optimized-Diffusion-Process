import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

def imshow_step(title, image, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def scan_image(image):
    # Create a set to store unique pixel values
    unique_pixels = set()

    # Iterate over each pixel in the image
    for row in image:
        for pixel in row:
            # Check if the pixel value is not 0 and not already in the set
            if pixel.any() and tuple(pixel) not in unique_pixels:
                unique_pixels.add(tuple(pixel))
                print(pixel)

def get_img_agnostic(img, parse, pose_data, output_path_agnostic, output_path_mask, im_name):
    parse_array = np.array(parse)
    parse_head = ((parse_array == 4).astype(np.float32) +
                  (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                   (parse_array == 12).astype(np.float32) +
                   (parse_array == 16).astype(np.float32) +
                   (parse_array == 17).astype(np.float32) +
                   (parse_array == 18).astype(np.float32) +
                   (parse_array == 19).astype(np.float32))

    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    mask = Image.new('L', img.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])
    point = (pose_data[9] + pose_data[12]) / 2
    pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
    r = int(length_a / 16) + 1
    
    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
    mask_draw.line([tuple(pose_data[i]) for i in [2, 5]], 255, width=r*10)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        mask_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 255, 255)
    for i in [3, 4, 6, 7]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
        mask_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 255, width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        mask_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 255, 255)
    
    # mask torso
    for i in [9, 12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        mask_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 255, 255)
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
    mask_draw.line([tuple(pose_data[i]) for i in [2, 9]], 255, width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
    mask_draw.line([tuple(pose_data[i]) for i in [5, 12]], 255, width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
    mask_draw.line([tuple(pose_data[i]) for i in [9, 12]], 255, width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')
    mask_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 255, 255)

    # mask neck
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
    mask_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 255, 255)

    # overlay head and lower body from original image
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

    # Save agnostic and mask images
    agnostic.save(osp.join(output_path_agnostic, im_name))
    mask.save(osp.join(output_path_mask, im_name.replace('.jpg', '_mask.png')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="dataset dir")

    args = parser.parse_args()
    data_path = args.data_path
    output_path = data_path

    output_path_agnostic = osp.join(output_path, 'agnostic-v3.2')
    output_path_mask = osp.join(output_path, 'agnostic-mask')
    
    os.makedirs(output_path_agnostic, exist_ok=True)
    os.makedirs(output_path_mask, exist_ok=True)
    
    for im_name in tqdm(os.listdir(osp.join(data_path, 'image'))):
        
        # load pose image
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        
        try:
            with open(osp.join(data_path, 'openpose_json', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:, :2]
        except IndexError:
            print(pose_name)
            continue

        # load parsing image
        im = Image.open(osp.join(data_path, 'image', im_name))
        label_name = im_name.replace('.jpg', '.png')
        im_label = Image.open(osp.join(data_path, 'image-parse-v3', label_name))
        get_img_agnostic(im, im_label, pose_data, output_path_agnostic, output_path_mask, im_name)