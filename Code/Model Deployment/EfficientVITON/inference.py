import os
from os.path import join as opj
from omegaconf import OmegaConf
from importlib import import_module
import base64

import cv2
import numpy as np
import torch

from cldm.plms_hacked import PLMSSampler
from cldm.model import create_model
from utils import tensor2img
from torch.utils.data import DataLoader

# --- Model Initialization (run only once)
config_path = "./configs/VITONHD.yaml"  # Fixed path
model_load_path = "./ckpts/VITONHD.ckpt" # Fixed path
data_root_dir = "./DATA/zalando-hd-resized"
save_dir = "./samples"
batch_size = 1
unpair = True
denoise_steps = 50
img_H = 512
img_W = 384
eta = 0.0

class Model:
    def __init__(self):
        config = OmegaConf.load(config_path)
        config.model.params.img_H = img_H
        config.model.params.img_W = img_W
        self.params = config.model.params
        self.config = config

    def initialize_model(self):
        model = create_model(config_path=None, config=self.config)
        load_cp = torch.load(model_load_path, map_location="cpu")
        load_cp = load_cp["state_dict"] if "state_dict" in load_cp.keys() else load_cp
        model.load_state_dict(load_cp)
        model = model.cuda()
        model.eval()

        self.sampler = PLMSSampler(model)

        self.shape = (4, img_H//8, img_W//8) 
        os.makedirs(save_dir, exist_ok=True)

        self.model = model

    @torch.no_grad()
    def run_inference(self):
        # Modify test_pairs.txt
        dataset_name = self.config.dataset_name
        dataset = getattr(import_module("dataset"), dataset_name)(
            data_root_dir=data_root_dir,
            img_H=img_H,
            img_W=img_W,
            is_paired=not unpair,
            is_test=True,
            is_sorted=True
        )

        dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=batch_size, pin_memory=True)

        for batch_idx, batch in enumerate(dataloader):
            print(f"{batch_idx}/{len(dataloader)}")
            z, c = self.model.get_input(batch, self.params.first_stage_key)
            bs = z.shape[0]
            c_crossattn = c["c_crossattn"][0][:bs]
            if c_crossattn.ndim == 4:
                c_crossattn = self.model.get_learned_conditioning(c_crossattn)
                c["c_crossattn"] = [c_crossattn]
            uc_cross = self.model.get_unconditional_conditioning(bs)
            uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
            uc_full["first_stage_cond"] = c["first_stage_cond"]
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            self.sampler.model.batch = batch

            ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
            start_code = self.model.q_sample(z, ts)

            samples, _, _ = self.sampler.sample(
                denoise_steps,
                bs,
                self.shape, 
                c,
                x_T=start_code,
                verbose=False,
                eta=eta,
                unconditional_conditioning=uc_full,
            )

            x_samples = self.model.decode_first_stage(samples)
            for sample_idx, (x_sample, fn,  cloth_fn) in enumerate(zip(x_samples, batch['img_fn'], batch["cloth_fn"])):
                x_sample_img = tensor2img(x_sample)  # [0, 255]
                # to_path = opj(save_dir, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg")
                # cv2.imwrite(to_path, x_sample_img[:,:,::-1])

                # Encode image to base64
                _, encoded_image = cv2.imencode('.jpg', x_sample_img[:,:,::-1])
                encoded_string = base64.b64encode(encoded_image).decode('utf-8')
                return encoded_string
# if __name__ == "__main__":
#     print("This script is intended to be used by the Flask API")
#     # Example Usage (This will be skipped if not main execution)
#     cloth_filename = "125"  # Replace with any cloth file name in cloth directory
#     result = run_inference(cloth_filename)
#     print(f"Result image saved to: {result}")