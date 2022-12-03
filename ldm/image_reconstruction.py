import argparse
import copy
import os
from pathlib import Path

from PIL import Image
import lpips
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from general_utils.seamless_cloning import poisson_seamless_clone
from omegaconf import OmegaConf
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.utils import make_grid

from ldm.image_editor import load_model_from_config, read_image, read_mask
from ldm.models.diffusion.ddpm import LatentDiffusion


class ImagesDataset(Dataset):
    def __init__(self, source_path, transform, indices=None):
        self.source_path = Path(source_path)
        self.img_names = os.listdir(source_path)
        self.img_names.sort()
        if indices is not None:
            self.img_names = [self.img_names[i] for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image = Image.open(self.source_path / self.img_names[idx]).convert("RGB")
        tensor_image = self.transform(image)
        tensor_image = tensor_image * 2.0 - 1.0

        return tensor_image


class ImageReconstruction:
    def __init__(
        self,
        verbose: bool = False,
    ):
        self.opt = self.get_arguments()
        config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
        self.device = (
            torch.device(f"cuda:{self.opt.gpu_id}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model = load_model_from_config(
            config=config, ckpt="models/ldm/text2img-large/model.ckpt", device=self.device
        )
        self.model = self.model.to(self.device)

        img_size = (self.opt.W, self.opt.H)
        mask_size = (self.opt.W // 8, self.opt.H // 8)
        self.init_image = read_image(
            img_path=self.opt.init_image, device=self.device, dest_size=img_size
        )
        self.mask, self.org_mask = read_mask(
            mask_path=self.opt.mask, device=self.device, dest_size=mask_size, img_size=img_size
        )
        if self.opt.invert_mask:
            self.mask = 1 - self.mask
            self.org_mask = 1 - self.org_mask

        self.verbose = verbose
        # self.lpips_model = lpips.LPIPS(net="vgg").to(model.device)

        samples_dataset = ImagesDataset(
            source_path=os.path.join(self.opt.images_path, "images"),
            transform=ToTensor(),
            indices=self.opt.selected_indices,
        )

        reconstructed_samples = self._reconstruct_background(samples_dataset)
        self._save_visualization(reconstructed_samples)

    def get_arguments(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--init_image", type=str, default="", help="a source image to edit")
        parser.add_argument("--mask", type=str, default="", help="a mask to edit the image")
        parser.add_argument(
            "--invert_mask",
            help="Indicator enabling inverting the input mask",
            action="store_true",
            dest="invert_mask",
        )
        parser.add_argument(
            "--images_path",
            type=str,
            default="outputs/edit_results/samples/",
            help="The path for the images to reconstruct",
        )
        parser.add_argument(
            "--H",
            type=int,
            default=256,
            help="image height, in pixel space",
        )

        parser.add_argument(
            "--W",
            type=int,
            default=256,
            help="image width, in pixel space",
        )

        parser.add_argument(
            "--batch_size",
            type=int,
            default=16,
            help="The number of optimization steps in case of optimization",
        )

        parser.add_argument(
            "--optimization_steps",
            type=int,
            default=75,
            help="The number of optimization steps in case of optimization",
        )

        parser.add_argument(
            "--reconstruction_type",
            type=str,
            help="The background reconstruction type",
            default="optimization",
            choices=["optimization", "pixel", "poisson"],
        )
        parser.add_argument(
            "--optimization_mode",
            type=str,
            help="The optimization mode in case of optimization reconstruction type",
            default="weights",
            choices=["weights", "latents"],
        )
        parser.add_argument(
            "--selected_indices",
            type=int,
            nargs="+",
            default=None,
            help="The indices to reconstruct, if not given - will reconstruct all the images",
        )

        # Misc
        parser.add_argument(
            "--gpu_id",
            type=int,
            default=0,
            help="The GPU specific id",
        )

        opt = parser.parse_args()

        return opt

    def _reconstruct_background(self, samples):
        reconstructed_samples = []

        if self.opt.reconstruction_type == "pixel":
            for sample in samples:
                sample = sample.to(self.device) * self.org_mask[0] + self.init_image * (
                    1 - self.org_mask[0]
                )
                sample = torch.clamp((sample + 1.0) / 2.0, min=0.0, max=1.0)
                reconstructed_samples.append(sample)
        elif self.opt.reconstruction_type == "poisson":
            mask_numpy = self.org_mask.squeeze().cpu().numpy()
            init_image_numpy = rearrange(
                ((self.init_image + 1) / 2).squeeze().cpu().numpy(), "c h w -> h w c"
            )

            for sample in samples:
                sample = torch.clamp((sample + 1.0) / 2.0, min=0.0, max=1.0)
                curr_sample = rearrange(sample.cpu().numpy(), "c h w -> h w c")
                cloned_sample = poisson_seamless_clone(
                    source_image=curr_sample,
                    destination_image=init_image_numpy,
                    mask=mask_numpy,
                )
                cloned_sample = torch.from_numpy(
                    cloned_sample[np.newaxis, ...].transpose(0, 3, 1, 2)
                ).to(self.device)
                reconstructed_samples.append(cloned_sample)
        elif self.opt.reconstruction_type == "optimization":
            for sample in samples:
                optimized_sample = self.reconstruct_image_by_optimization(
                    fg_image=sample.to(self.device).unsqueeze(0),
                    bg_image=self.init_image,
                    mask=self.org_mask,
                )
                optimized_sample = torch.clamp(optimized_sample, min=0.0, max=1.0)
                reconstructed_samples.append(optimized_sample)
        else:
            raise ValueError("Missing reconstruction type")

        reconstructed_samples = torch.cat(reconstructed_samples)
        return reconstructed_samples

    def loss(
        self,
        fg_image: torch.Tensor,
        bg_image: torch.Tensor,
        curr_latent: torch.Tensor,
        mask: torch.Tensor,
        preservation_ratio: float = 100,
    ):
        curr_reconstruction = self.model.decode_first_stage(curr_latent)
        loss = (
            F.mse_loss(fg_image * mask, curr_reconstruction * mask)
            + F.mse_loss(bg_image * (1 - mask), curr_reconstruction * (1 - mask))
            * preservation_ratio
        )
        # loss = self.lpips_model(fg_image * mask, curr_reconstruction * mask).sum() + \
        #     self.lpips_model(bg_image * (1 - mask), curr_reconstruction * (1 - mask)).sum()

        return loss

    @torch.no_grad()
    def get_curr_reconstruction(self, curr_latent):
        curr_reconstruction = self.model.decode_first_stage(curr_latent)
        curr_reconstruction = torch.clamp((curr_reconstruction + 1.0) / 2.0, min=0.0, max=1.0)

        return curr_reconstruction

    @torch.no_grad()
    def plot_reconstructed_image(self, curr_latent, fg_image, bg_image, mask):
        curr_reconstruction = self.get_curr_reconstruction(curr_latent=curr_latent)
        curr_reconstruction = curr_reconstruction[0].cpu().numpy().transpose(1, 2, 0)

        fg_image = torch.clamp((fg_image + 1.0) / 2.0, min=0.0, max=1.0)
        fg_image = fg_image[0].cpu().numpy().transpose(1, 2, 0)

        bg_image = torch.clamp((bg_image + 1.0) / 2.0, min=0.0, max=1.0)
        bg_image = bg_image[0].cpu().numpy().transpose(1, 2, 0)

        mask = mask[0].detach().cpu().numpy().transpose(1, 2, 0)
        composed = fg_image * mask + bg_image * (1 - mask)

        plt.imshow(np.hstack([bg_image, fg_image, composed, curr_reconstruction]))
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def _save_visualization(self, samples, images_per_row: int = 6):
        self._save_images(samples)

        # Add source image and mask to visualization
        if self.init_image is not None:
            blank_image = torch.ones_like(self.init_image)
            if self.mask is None:
                self.org_mask = blank_image
                resized_mask = blank_image
            else:
                self.org_mask = self.org_mask.repeat((1, 3, 1, 1))

                resized_mask = F.interpolate(self.mask, size=(self.opt.H, self.opt.W))
                resized_mask = resized_mask.repeat((1, 3, 1, 1))

            encoder_posterior = self.model.encode_first_stage(self.init_image)
            encoder_result = self.model.get_first_stage_encoding(encoder_posterior)
            reconstructed_image = self.model.decode_first_stage(encoder_result)
            reconstructed_image = torch.clamp((reconstructed_image + 1.0) / 2.0, min=0.0, max=1.0)

            inputs_row = [
                (self.init_image + 1) / 2,
                reconstructed_image,
                self.org_mask,
                resized_mask,
            ]
            pad_row = [blank_image for _ in range(images_per_row - len(inputs_row))]
            inputs_row = inputs_row + pad_row

            samples = torch.cat([torch.cat(inputs_row), samples])

        grid = make_grid(samples, nrow=images_per_row)
        grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
        Image.fromarray(grid.astype(np.uint8)).save(
            os.path.join(self.opt.images_path, f"reconstructed_{self.opt.reconstruction_type}.png")
        )

    def _save_images(self, samples):
        samples_dir = os.path.join(
            self.opt.images_path,
            f"reconstructed_{self.opt.reconstruction_type}",
        )
        os.makedirs(samples_dir, exist_ok=True)
        for i, sample in enumerate(samples):
            sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
            Image.fromarray(sample.astype(np.uint8)).save(os.path.join(samples_dir, f"{i:04}.png"))

    def reconstruct_image_by_optimization(
        self, fg_image: torch.Tensor, bg_image: torch.Tensor, mask: torch.Tensor
    ):
        encoder_posterior = self.model.encode_first_stage(fg_image)
        initial_latent = self.model.get_first_stage_encoding(encoder_posterior)

        if self.opt.optimization_mode == "weights":
            curr_latent = initial_latent.clone().detach()
            decoder_copy = copy.deepcopy(self.model.first_stage_model.decoder)
            self.model.first_stage_model.decoder.requires_grad_(True)
            optimizer = optim.Adam(self.model.first_stage_model.decoder.parameters(), lr=0.0001)
        else:
            curr_latent = initial_latent.clone().detach().requires_grad_(True)
            optimizer = optim.Adam([curr_latent], lr=0.1)

        for i in tqdm(range(self.opt.optimization_steps), desc="Reconstruction optimization"):
            if self.verbose and i % 25 == 0:
                self.plot_reconstructed_image(
                    curr_latent=curr_latent,
                    fg_image=fg_image,
                    bg_image=bg_image,
                    mask=mask,
                )
            optimizer.zero_grad()

            loss = self.loss(
                fg_image=fg_image, bg_image=bg_image, curr_latent=curr_latent, mask=mask
            )

            if self.verbose:
                print(f"Iteration {i}: Curr loss is {loss}")

            loss.backward()
            optimizer.step()

        reconstructed_result = self.get_curr_reconstruction(curr_latent=curr_latent)

        if self.opt.optimization_mode == "weights":
            self.model.first_stage_model.decoder = None
            self.model.first_stage_model.decoder = decoder_copy

        return reconstructed_result
