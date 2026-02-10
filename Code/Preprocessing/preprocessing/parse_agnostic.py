import json
from os import path as osp
import os

import numpy as np
from PIL import Image, ImageDraw

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def pad_palette_image(img, target_width, target_height):
    # Ensure the image is in mode 'P'
    if img.mode != 'P':
        raise ValueError("Image must be in mode 'P'")
    
    # Calculate the padding needed on each side
    pad_left = (target_width - img.width) // 2
    pad_top = (target_height - img.height) // 2
    pad_right = target_width - img.width - pad_left
    pad_bottom = target_height - img.height - pad_top

    # Create a new image with the target size and zero (black) pixels
    padded_img = Image.new('P', (target_width, target_height), color=0)
    
    # Ensure the palette of the new image matches the original image
    padded_img.putpalette(img.getpalette())
    
    # Paste the original image into the center of the new image
    padded_img.paste(img, (pad_left, pad_top))
    
    return padded_img


def imshow_step(title, image, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def get_im_parse_agnostic(im_parse, pose_data, w=768, h=1024):
    # pad the im_parse to be the same size as the original image
    original_width, original_height = im_parse.size
    im_parse = pad_palette_image(im_parse, w, h)

    parse_array = np.array(im_parse)
    parse_upper = ((parse_array == 5).astype(np.float32) +
                   (parse_array == 6).astype(np.float32) +
                   (parse_array == 7).astype(np.float32))
    parse_neck = (parse_array == 10).astype(np.float32)

    r = 10
    agnostic = im_parse.copy()

    # imshow_step("Original Parse", im_parse)

    # mask arms
    for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
        mask_arm = Image.new('L', (w, h), 'black')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        i_prev = pose_ids[0]
        for i in pose_ids[1:]:
            if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r * 10)
            pointx, pointy = pose_data[i]
            radius = r * 4 if i == pose_ids[-1] else r * 15
            mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
            i_prev = i
        parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        # imshow_step(f"Mask Arm for parse_id {parse_id}", np.uint8(parse_arm * 255))

    # mask torso & neck
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    # imshow_step("Masked Upper Body", np.uint8(parse_upper * 255))

    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))
    # imshow_step("Masked Neck", np.uint8(parse_neck * 255))

    # Crop the image back to its original size
    pad_left = (w - original_width) // 2
    pad_top = (h - original_height) // 2
    agnostic = agnostic.crop((pad_left, pad_top, pad_left + original_width, pad_top + original_height))
    # imshow_step("Cropped to Original Size", agnostic)

    return agnostic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="dataset dir")

    args = parser.parse_args()
    data_path = args.data_path
    output_path = data_path
    output_path = osp.join(output_path, "image-parse-agnostic-v3.2")

    os.makedirs(output_path, exist_ok=True)

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
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(data_path, 'image-parse-v3', parse_name))

        agnostic = get_im_parse_agnostic(im_parse, pose_data)

        # imshow_step("Final Agnostic", agnostic)
        agnostic.save(osp.join(output_path, parse_name))