import argparse
import os
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch import nn
from einops import rearrange
from general_utils.seamless_cloning import poisson_seamless_clone
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from scipy.ndimage import binary_dilation

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion

from ldm.util import instantiate_from_config
import clip


def load_model_from_config(config, ckpt, device, verbose=False) -> LatentDiffusion:
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.cond_stage_model.device = device
    model.cond_stage_model.tknz_fn.device = device
    model.eval()
    return model


def read_image(img_path: str, device, dest_size=(256, 256)):
    image = Image.open(img_path).convert("RGB")
    image = image.resize(dest_size, Image.LANCZOS)
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(device)

    image = image * 2.0 - 1.0

    return image


def read_mask(
    mask_path: str, device, dilation_iterations: int = 0, dest_size=(32, 32), img_size=(256, 256)
):
    org_mask = Image.open(mask_path).convert("L")
    mask = org_mask.resize(dest_size, Image.NEAREST)
    mask = np.array(mask) / 255

    masks_array = []
    for i in reversed(range(dilation_iterations)):
        k_size = 3 + 2 * i
        masks_array.append(binary_dilation(mask, structure=np.ones((k_size, k_size))))
    masks_array.append(mask)
    masks_array = np.array(masks_array).astype(np.float32)
    masks_array = masks_array[:, np.newaxis, :]
    masks_array = torch.from_numpy(masks_array).to(device)

    org_mask = org_mask.resize(img_size, Image.LANCZOS)
    org_mask = np.array(org_mask).astype(np.float32) / 255.0
    org_mask = org_mask[None, None]
    org_mask[org_mask < 0.5] = 0
    org_mask[org_mask >= 0.5] = 1
    org_mask = torch.from_numpy(org_mask).to(device)

    return masks_array, org_mask


def normalized_cosine_distance(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    distance = 1 - (x @ y.t()).squeeze()

    return distance


class ImageEditor:
    def get_arguments(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--prompt",
            type=str,
            nargs="?",
            default="a painting of a virus monster playing guitar",
            help="the prompt to render",
        )

        parser.add_argument(
            "--outdir",
            type=str,
            nargs="?",
            help="dir to write results to",
            default="outputs/edit_results",
        )
        parser.add_argument(
            "--ddim_steps",
            type=int,
            default=50,
            help="number of ddim sampling steps",
        )
        parser.add_argument(
            "--skip_steps",
            type=int,
            default=0,
            help="the number of diffusion steps to skip (out of ddim_steps)",
        )
        parser.add_argument(
            "--percentage_of_pixel_blending",
            type=float,
            default=0,
            help="The percentage of steps to perform diffusion in the pixel-level",
        )

        parser.add_argument(
            "--ddim_eta",
            type=float,
            default=0.0,
            help="ddim eta (eta=0.0 corresponds to deterministic sampling",
        )
        parser.add_argument(
            "--n_iter",
            type=int,
            default=1,
            help="sample this often",
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
            "--n_samples",
            type=int,
            default=4,
            help="how many samples to produce for the given prompt",
        )

        parser.add_argument(
            "--scale",
            type=float,
            default=5.0,
            help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        )

        parser.add_argument("--init_image", type=str, default="", help="a source image to edit")
        parser.add_argument("--mask", type=str, default="", help="a mask to edit the image")
        parser.add_argument(
            "--invert_mask",
            help="Indicator enabling inverting the input mask",
            action="store_true",
            dest="invert_mask",
        )
        parser.add_argument(
            "--mask_dilation_iterations",
            type=int,
            default=0,
            help="How many times to dilate the mask",
        )
        parser.add_argument(
            "--save_video",
            help="Indicator for saving the diffusion pred_x0 vide",
            action="store_true",
            dest="save_video",
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

    def _get_samples(self):
        all_samples = list()
        with self.model.ema_scope():
            uc = None
            if self.opt.scale != 1.0:
                uc = self.model.get_learned_conditioning(self.opt.n_samples * [""])
            for n in trange(self.opt.n_iter, desc="Sampling"):
                c = self.model.get_learned_conditioning(self.opt.n_samples * [self.opt.prompt])
                shape = [4, self.opt.H // 8, self.opt.W // 8]
                samples_ddim, intermediates = self.sampler.sample(
                    S=self.opt.ddim_steps,
                    conditioning=c,
                    batch_size=self.opt.n_samples,
                    shape=shape,
                    verbose=False,
                    log_every_t=1 if self.opt.save_video else 50,
                    unconditional_guidance_scale=self.opt.scale,
                    unconditional_conditioning=uc,
                    eta=self.opt.ddim_eta,
                    skip_steps=self.opt.skip_steps,
                    init_image=self.init_image,
                    mask=self.mask,
                    org_mask=self.org_mask,
                    percentage_of_pixel_blending=self.opt.percentage_of_pixel_blending,
                )

                if self.opt.save_video:
                    self._save_video(
                        sample_path=self.sample_path,
                        intermediates=intermediates,
                    )

                x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                all_samples.append(x_samples_ddim)

        return all_samples

    def _save_images(self, samples, samples_path):
        for i, sample in enumerate(samples):
            sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
            Image.fromarray(sample.astype(np.uint8)).save(os.path.join(samples_path, f"{i:04}.png"))

    @torch.no_grad()
    def _save_video(self, sample_path, intermediates, fps: float = 1.5):
        intermediates_pred_x0 = intermediates["pred_x0"]
        batch_size = len(intermediates_pred_x0[0])
        frames_per_sample = [[] for _ in range(batch_size)]

        for intermediate in tqdm(intermediates_pred_x0, desc="Saving video"):
            x = self.model.decode_first_stage(intermediate)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x.cpu().numpy(), "b c h w -> b h w c")
            x = x.astype(np.uint8)
            for i in range(len(x)):
                frames_per_sample[i].append(x[i])

        height, width, _ = frames_per_sample[0][0].shape
        for i, frames in enumerate(frames_per_sample):
            video = cv2.VideoWriter(
                str(sample_path / "videos" / f"{i:04}.mp4"),
                apiPreference=0,
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                fps=fps,
                frameSize=(width, height),
            )

            for frame in frames:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            cv2.destroyAllWindows()
            video.release()

    def _save_visualization(self, all_samples, images_per_row: int = 6):
        all_samples = torch.cat(all_samples)
        if self.opt.n_samples > 1 and not self.opt.save_video:
            all_samples = self._get_sorted_results_by_CLIP(all_samples)
        self._save_images(all_samples, os.path.join(self.sample_path / "images"))

        # Add source image and mask to visualization
        if self.init_image is not None:
            blank_image = torch.ones_like(self.init_image)
            if self.mask is None:
                self.org_mask = blank_image
                resized_mask = blank_image
            else:
                self.org_mask = self.org_mask.repeat((1, 3, 1, 1))

                resized_mask = F.interpolate(
                    self.mask[-1].unsqueeze(0), size=(self.opt.H, self.opt.W)
                )
                resized_mask = resized_mask.repeat((1, 3, 1, 1))

            encoder_posterior = self.model.encode_first_stage(self.init_image)
            encoder_result = self.model.get_first_stage_encoding(encoder_posterior)
            reconstructed_image = self.model.decode_first_stage(encoder_result)
            reconstructed_image = torch.clamp((reconstructed_image + 1.0) / 2.0, min=0.0, max=1.0)

            init_image = (self.init_image + 1) / 2
            self._save_images(
                [init_image[0], self.org_mask[0]], os.path.join(self.sample_path / "assets")
            )
            inputs_row = [
                init_image,
                reconstructed_image,
                self.org_mask,
                resized_mask,
            ]
            pad_row = [blank_image for _ in range(images_per_row - len(inputs_row))]
            inputs_row = inputs_row + pad_row

            all_samples = torch.cat([torch.cat(inputs_row), all_samples])

        grid = make_grid(all_samples, nrow=images_per_row)

        # to image
        grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
        Image.fromarray(grid.astype(np.uint8)).save(
            os.path.join(self.opt.outdir, f'{self.opt.prompt.replace(" ", "-")}.png')
        )
        print(f"Saved to: {self.opt.outdir}")

    @torch.no_grad()
    def _get_sorted_results_by_CLIP(self, samples):
        clip_model = (
            clip.load("ViT-B/16", device=self.device, jit=False)[0].eval().requires_grad_(False)
        )
        clip_size = clip_model.visual.input_resolution
        clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )

        text_embed = clip_model.encode_text(clip.tokenize(self.opt.prompt).to(self.device)).float()

        distances = []
        for samples_batch in torch.split(samples, self.opt.n_samples):
            masked_samples = samples_batch * self.org_mask
            clip_samples = TF.resize(masked_samples, [clip_size, clip_size])
            clip_samples = clip_normalize(clip_samples)
            image_embeds = clip_model.encode_image(clip_samples).float()

            curr_distances = normalized_cosine_distance(image_embeds, text_embed)
            distances.append(curr_distances)
        distances = torch.cat(distances)

        argsort_distances = torch.argsort(distances)
        sorted_samples = samples[argsort_distances]

        return sorted_samples

    def edit_image(self):
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
        self.sampler = DDIMSampler(self.model)

        os.makedirs(self.opt.outdir, exist_ok=True)
        self.sample_path = Path(self.opt.outdir) / "samples"
        os.makedirs(self.sample_path / "images", exist_ok=True)
        os.makedirs(self.sample_path / "assets", exist_ok=True)
        if self.opt.save_video:
            os.makedirs(self.sample_path / "videos", exist_ok=True)

        img_size = (self.opt.W, self.opt.H)
        mask_size = (self.opt.W // 8, self.opt.H // 8)
        self.init_image = read_image(
            img_path=self.opt.init_image, device=self.device, dest_size=img_size
        )

        if self.opt.mask == "":
            self.mask, self.org_mask = None, None
        else:
            self.mask, self.org_mask = read_mask(
                mask_path=self.opt.mask,
                dilation_iterations=self.opt.mask_dilation_iterations,
                device=self.device,
                dest_size=mask_size,
                img_size=img_size,
            )
            if self.opt.invert_mask:
                self.mask = 1 - self.mask
                self.org_mask = 1 - self.org_mask

        all_samples = self._get_samples()
        self._save_visualization(all_samples)
