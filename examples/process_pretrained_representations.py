import os 
import numpy as np 
from glob import glob 
import torch
from torchvision.io import read_image
from tqdm import tqdm, trange

import voltron
from voltron import instantiate_extractor, load
import cv2 

import pickle


# INPUT_DATADIR = "/iris/u/khatch/vd5rl/datasets/diversekitchen/expert_demos"
# OUTPUT_DATADIR = "/iris/u/khatch/vd5rl/datasets/diversekitchen_v-cond/expert_demos"

# INPUT_DATADIR = "/iris/u/khatch/vd5rl/datasets/diversekitchen/play_data"
INPUT_DATADIR = "/scr-ssd/khatch/diversekitchen/play_data"
OUTPUT_DATADIR = "/iris/u/khatch/vd5rl/datasets/diversekitchen_v-cond/play_data"
ENCODER_MODEL = "v-cond"

CAMERAS = ['camera_0', 'camera_1', 'camera_gripper']

class PretrainedEncoder:
    def __init__(self, model_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vcond, preprocess = load(model_name, device=device, freeze=True)

        img = preprocess(read_image("peel-carrot-initial.png"))[None, ...].to(device)
        print("img.shape:", img.shape)
        

        with torch.no_grad():
            if "v-cond" in model_name:
                visual_features = vcond(img, mode="visual")  # Vision-only features (no language)
            else:
                visual_features = vcond(img)  # Vision-only features (no language)

        vector_extractor = instantiate_extractor(vcond, n_latents=1)().to(device)
        print("vector_extractor(visual_features).shape:", vector_extractor(visual_features).shape)

        self._vcond = vcond
        self._preprocess = preprocess
        self._imsize = 224
        self._model_name = model_name
        self._vector_extractor = vector_extractor
        self._device = device
        self._embed_dim = vector_extractor(visual_features).squeeze().shape[0] * 3


    def __call__(self, pixels):
        pixels = pixels.copy()

        if len(pixels.shape) == 3:

            pixels = np.stack([pixels[..., :3], pixels[..., 3:6], pixels[..., 6:]])

            assert len(pixels.shape) == 4, f"pixels.shape: {pixels.shape}"

            pixels = pixels.transpose((0, 3, 1, 2))
            pixels = torch.tensor(pixels, device=self._device)
            
            img = self._preprocess(pixels).to(self._device)

            with torch.no_grad():
                if "v-cond" in self._model_name:
                    visual_features = self._vcond(img, mode="visual")  # Vision-only features (no language)
                else:
                    visual_features = self._vcond(img)  # Vision-only features (no language)
            
            features = self._vector_extractor(visual_features) # (batch, 384)
            features = features.view(-1)
            
        elif len(pixels.shape) == 4:
            
            pixels = np.stack([pixels[..., :3], pixels[..., 3:6], pixels[..., 6:]], axis=0)

            N = pixels.shape[1]

            # features = torch.zeros((N, self._embed_dim)).to(self._device)
            features = torch.zeros((N, 3, self._embed_dim // 3)).to(self._device)
            
            for cam_i in range(pixels.shape[0]):
                pixels_cam_i =  pixels[cam_i].copy()
                pixels_cam_i = pixels_cam_i.transpose((0, 3, 1, 2))
                assert len(pixels_cam_i.shape) == 4, f"pixels_cam_i.shape: {pixels_cam_i.shape}"
                pixels_cam_i = torch.tensor(pixels_cam_i, device=self._device)
                img_cam_i = self._preprocess(pixels_cam_i).to(self._device)

                with torch.no_grad():
                    if "v-cond" in self._model_name:
                        visual_features_cam_i = self._vcond(img_cam_i, mode="visual")  # Vision-only features (no language)
                    else:
                        visual_features_cam_i = self._vcond(img_cam_i)  # Vision-only features (no language)
                        
                    features_cam_i = self._vector_extractor(visual_features_cam_i) # (batch, 384)

                    # start = cam_i*(self._embed_dim // 3)
                    # end = start + (self._embed_dim // 3)
                    # print(f"[{start}:{end}]")
                    # features[:, start:end] = features_cam_i


                    # print("cam_i:", cam_i)
                    # print("features_cam_i.shape:", features_cam_i.shape)
                    # print(f"features[:, {cam_i}].shape:", features[:, cam_i].shape)
                    # print(f"features.shape:", features.shape)

                    features[:, cam_i] = features_cam_i

            features = features.view(N, -1)
            assert features.shape[-1] == self._embed_dim
        else:
            raise ValueError(f"pixels.shape: {pixels.shape}")

        return features.detach().cpu().numpy()




def get_representations():
    pretrained_encoder = PretrainedEncoder(ENCODER_MODEL)    

    os.makedirs(OUTPUT_DATADIR, exist_ok=True)

    episode_files = glob(os.path.join(INPUT_DATADIR, "*.pkl"))

    # completed_episode_files = glob(os.path.join(OUTPUT_DATADIR, "*.pkl"))
    # completed_episode_names = set([episode_file.split("/")[-1] for episode_file in completed_episode_files])

    for episode_file in tqdm(episode_files):
        
        episode_name = episode_file.split("/")[-1]
        new_episode_file = os.path.join(OUTPUT_DATADIR, episode_name)

        if os.path.isfile(new_episode_file):
            print(f"Skipping \"{new_episode_file}\" because it already exists...")
            continue 

        episode = np.load(episode_file, allow_pickle=True)

        allcams_images = np.concatenate([episode[cam + '_rgb'] for cam in CAMERAS], axis=-1)

        # pretrained_representations = np.zeros((allcams_images.shape[0], pretrained_encoder._embed_dim))
        # for t in range(allcams_images.shape[0]):
        #     pretrained_representations[t] = pretrained_encoder(allcams_images[t])

        pretrained_representations = pretrained_encoder(allcams_images)
        episode["pretrained_representations"] = pretrained_representations

        with open(new_episode_file, 'wb') as f:
            pickle.dump(episode, f, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":
    get_representations()

