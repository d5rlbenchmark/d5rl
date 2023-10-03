"""
usage.py

Example script demonstrating how to load a Voltron model (`V-Cond`) and instantiate a Multiheaded Attention Pooling
extractor head for downstream tasks.

This is the basic formula/protocol for using Voltron for arbitrary downstream applications.

Run with (from root of repository): `python examples/usage.py`
"""
import torch
from torchvision.io import read_image

from voltron import instantiate_extractor, load


def usage() -> None:
    print("[*] Demonstrating Voltron Usage for Various Adaptation Applications")

    # Get `torch.device` for loading model (note -- we'll load weights directly onto device!)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Voltron model --> specify `freeze`, `device` and get model (nn.Module) and preprocessor
    vcond, preprocess = load("v-cond", device=device, freeze=True)
    # vcond, preprocess = load("v-gen", device=device, freeze=True)
    # vcond, preprocess = load("v-dual", device=device, freeze=True)
    # vcond, preprocess = load("r-r3m-vit", device=device, freeze=True)
    # vcond, preprocess = load("r-r3m-rn50", device=device, freeze=True)
    # vcond, preprocess = load("v-cond-base", device=device, freeze=True)
    # vcond, preprocess = load("r-mvp", device=device, freeze=True)

    # Obtain and preprocess an image =>> can be from a dataset, from a camera on a robot, etc.
    img = preprocess(read_image("peel-carrot-initial.png"))[None, ...].to(device)
    lang = ["peeling a carrot"]
    

    # Works: v-cond, r-r3m-vit, v-cond-base

    print("img.shape:", img.shape)

    # # Get various representations...
    # with torch.no_grad():
    #     multimodal_features = vcond(img, lang, mode="multimodal")  # Fused vision & language features
    #     visual_features = vcond(img, mode="visual")  # Vision-only features (no language)

    # # Can instantiate various extractors for downstream applications
    # # vector_extractor = instantiate_extractor(vcond, n_latents=1, device=device)()
    # # seq_extractor = instantiate_extractor(vcond, n_latents=64, device=device)()
    # vector_extractor = instantiate_extractor(vcond, n_latents=1)().to(device)
    # seq_extractor = instantiate_extractor(vcond, n_latents=64)().to(device)

    # # Assertions...
    # assert list(vector_extractor(multimodal_features).shape) == [1, vcond.embed_dim], "Should return a dense vector!"
    # assert list(seq_extractor(visual_features).shape) == [1, 64, vcond.embed_dim], "Should return a sequence!"

    # vector_extractor(visual_features)

    # Get various representations...
    with torch.no_grad():
        # visual_features = vcond(img, mode="visual")  # Vision-only features (no language)
        visual_features = vcond(img)  # Vision-only features (no language)

    vector_extractor = instantiate_extractor(vcond, n_latents=1)().to(device)


    # Assertions...
    # assert list(vector_extractor(multimodal_features).shape) == [1, vcond.embed_dim], "Should return a dense vector!"
    print("vector_extractor(visual_features).shape:", vector_extractor(visual_features).shape)

    # img2 = img[..., :128, :128]
    # img2 = img[..., :512, :512]
    # img2 = img[..., :224, :224]
    # img2 = img[..., :223, :223]
    # with torch.no_grad():
    #     visual_features2 = vcond(img2, mode="visual")  # Vision-only features (no language)
    #     vector_extractor(visual_features2).shape

    # r3m, preprocess_r3m = load("r-r3m-vit", device=device, freeze=True)
    # img_r3m = preprocess_r3m(read_image("peel-carrot-initial.png"))[None, ...].to(device)
    # vector_extractor_r3m = instantiate_extractor(r3m, n_latents=1)().to(device)
    # visual_features_r3m = r3m(img_r3m)  # Vision-only features (no language)
    # vector_extractor_r3m(visual_features_r3m).shape


    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    usage()