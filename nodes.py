import numpy as np
import torch
import rembg
from PIL import Image

from nodes import VAELoader, VAEDecode


class vae_processor:
    LOADER = VAELoader()
    DECODER = VAEDecode()
    DEFAULT_VAE = "Baked VAE"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "baked_vae": ("VAE",),
                "vae_name": ([cls.DEFAULT_VAE] + cls.LOADER.vae_list(),),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "rcsaquino"

    def main(self, latent, baked_vae, vae_name):
        if vae_name == self.DEFAULT_VAE:
            return self.DECODER.decode(baked_vae, latent)
        vae = self.LOADER.load_vae(vae_name)[0]
        return self.DECODER.decode(vae, latent)


class vae_loader:
    LOADER = VAELoader()
    DEFAULT_VAE = "Baked VAE"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "baked_vae": ("VAE",),
                "vae_name": ([cls.DEFAULT_VAE] + cls.LOADER.vae_list(),),
            },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "main"
    CATEGORY = "rcsaquino"

    def main(self, baked_vae, vae_name):
        if vae_name == self.DEFAULT_VAE:
            return (baked_vae,)
        return self.LOADER.load_vae(vae_name)


class background_remover:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (
                    [
                        "u2net",
                        "u2netp",
                        "u2net_human_seg",
                        "u2net_cloth_seg",
                        "silueta",
                        "isnet-general-use",
                        "isnet-anime",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "rcsaquino"

    def main(self, image, model):
        # Referenced from https://github.com/Jcd1230/rembg-comfyui-node/blob/fac7df6c3f42e158a2829b511b37f13e4cc834eb/__init__.py#L31
        session = rembg.new_session(model)
        processed_img = rembg.remove(
            Image.fromarray(
                np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            ),
            session=session,
        )
        comfyui_img = torch.from_numpy(
            np.array(processed_img).astype(np.float32) / 255.0
        ).unsqueeze(0)
        return (comfyui_img,)


NODE_CLASS_MAPPINGS = {
    "VAEProcessor | rcsaquino": vae_processor,
    "VAELoader | rcsaquino": vae_loader,
    "BackgroundRemover | rcsaquino": background_remover,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEProcessor | rcsaquino": "VAE Processor",
    "VAELoader | rcsaquino": "VAE Loader",
    "BackgroundRemover | rcsaquino": "Background Remover",
}
