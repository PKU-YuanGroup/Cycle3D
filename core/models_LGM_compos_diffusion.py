import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import kiui
from kiui.lpips import LPIPS

from core.unet_LGM_compos import UNet
from core.options_latents_diffusion import Options
from core.gs import GaussianRenderer
from diffusers import AutoencoderKL, DDPMScheduler,  UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional
import random
import torchvision.transforms.functional as TF
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class LGM(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt

        # unet
        self.unet = UNet(
            9, 14, 
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
        )

        # last conv
        self.conv = nn.Conv2d(14, 14, kernel_size=1) # NOTE: maybe remove it if train again

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)
        
        model_key = opt.pretrained_model_name_or_path
        self.unet2 = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", low_cpu_mem_usage=False,device_map=None,ignore_mismatched_sizes=True)
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder")
        self.tokenizer =  CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.opt.weight_dtype)
        self.vae.requires_grad_(False)
        self.unet2.requires_grad_(False)
        #self.tokenizer.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

    
    def prepare_default_rays(self, device, elevation=0):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius),
            orbit_camera(elevation, 90, radius=self.opt.cam_radius),
            orbit_camera(elevation, 180, radius=self.opt.cam_radius),
            orbit_camera(elevation, 270, radius=self.opt.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure
            # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings
        

    def forward_gaussians(self, images, encoder_hidden_states, data):
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)
        timestep = data["timesteps"].flatten(0, 1)
        pred_noise, blocks_sample, temb= self.unet2(images, timestep, encoder_hidden_states, return_dict=False)
        
        pred_x0 = self.pred_x0(pred_noise, timestep, images)
        images_512 = (self.vae.decode(pred_x0.to(self.opt.weight_dtype) / 0.18215).sample +1)*0.5
        images_256 = F.interpolate(images_512.clamp(0, 1), (256, 256), mode='bilinear', align_corners=False)
        images_256 = TF.normalize(images_256, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        images_256 = torch.cat([images_256.to(self.opt.weight_dtype), data['ray'].flatten(0, 1).to(self.opt.weight_dtype) ], dim=1)

        x = self.unet(images_256, blocks_sample, temb) # [B*4, 14, h, w]
        x = self.conv(x) # [B*4, 14, h, w]

        x = x.reshape(B, 4, 14, self.opt.splat_size, self.opt.splat_size)
        
        ## visualize multi-view gaussian features for plotting figure
        # tmp_alpha = self.opacity_act(x[0, :, 3:4])
        # tmp_img_rgb = self.rgb_act(x[0, :, 11:]) * tmp_alpha + (1 - tmp_alpha)
        # tmp_img_pos = self.pos_act(x[0, :, 0:3]) * 0.5 + 0.5
        # kiui.vis.plot_image(tmp_img_rgb, save=True)
        # kiui.vis.plot_image(tmp_img_pos, save=True)

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
        
        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        
        return gaussians, images_512
    
    def pred_x0(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
        """
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=x.device)
        alpha_prod_t = alphas_cumprod [timestep]

        B = alpha_prod_t.shape[0]
        alpha_prod_t = alpha_prod_t.view(B, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        return pred_x0
    
    def encode_prompt(
            self,
            prompt,
            device,
            prompt_embeds: Optional[torch.FloatTensor] = None,
        ):
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            if prompt_embeds is None:
                text_inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = self.tokenizer.batch_decode(
                        untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                    )

                if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None

                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]

            if self.text_encoder is not None:
                prompt_embeds_dtype = self.text_encoder.dtype
            elif self.unet is not None:
                prompt_embeds_dtype = self.unet.dtype
            else:
                prompt_embeds_dtype = prompt_embeds.dtype

            prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            bs_embed, seq_len, _ = prompt_embeds.shape

            return prompt_embeds
    
    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr
    
    def forward(self, data, step_ratio=1):
        # data: output of the dataloader
        # return: loss

        results = {}
        loss = 0
        start_idx = None
        images = data['input'].to(self.opt.weight_dtype) # [B, 4, 9, h, W], input features
        
        num_views = images.shape[1]
        #ray_embedding = images[:, :, 4:]
        latents = images.flatten(0,1)
        latent = latents[:,:4]
        bsz, c, h, w = latent.shape
       
        # timesteps
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bsz // num_views,), device=images.device)
        timesteps_pred = timesteps.repeat_interleave(self.opt.num_views)
        timesteps = timesteps.repeat_interleave(num_views)
        timesteps = timesteps.long()
        if(random.random() < 0.7):
            start_idx = torch.randint(0,4, (1,)).item()
            timesteps[start_idx ::num_views] = 0
            timesteps_pred[start_idx ::self.opt.num_views] = 0
            
        if(random.random() < 0.7):
            prompt = data["prompt"]
            
            # prompt = [prompt[i][j] for j in range(len(prompt[0])) for i in range(len(prompt))]
            # encoder_hidden_states = self.encode_prompt(prompt, images.device).to(images.dtype)
            prompt = [prompt[0][i] for i in range(len(prompt[0]))]
            #print(prompt)
            encoder_hidden_states = self.encode_prompt(prompt, images.device).to(images.dtype)
            encoder_hidden_states = encoder_hidden_states[:,None].repeat(1,images.shape[1], 1, 1)
            encoder_hidden_states = encoder_hidden_states.flatten(0,1)
        else:
            prompt = ['']*images.shape[0]
            encoder_hidden_states = self.encode_prompt(prompt, images.device).to(images.dtype)
            encoder_hidden_states = encoder_hidden_states[:,None].repeat(1,images.shape[1], 1, 1)
            encoder_hidden_states = encoder_hidden_states.flatten(0,1)
        
        noise = torch.randn_like(latent).to(device=images.device)
        noisy_latents = self.scheduler.add_noise(latent, noise, timesteps).to(device=images.device)
        data['noisy_latents'] = noisy_latents.reshape(bsz // num_views, num_views, c, h, w)
        data['timesteps'] = timesteps.reshape(bsz // num_views, num_views)

        snr = self.compute_snr(timesteps_pred)
        mse_loss_weights = torch.stack([snr, self.opt.snr_gamma * torch.ones_like(timesteps_pred)], dim=1).min(dim=1)[0] 
        # use the first view to predict gaussians
        images = data['noisy_latents']
        gaussians, noise_images = self.forward_gaussians(images, encoder_hidden_states, data) # [B, N, 14]

        results['gaussians'] = gaussians

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
        
        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = results['image'].to(self.opt.weight_dtype) # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'].to(self.opt.weight_dtype) # [B, V, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas

        gt_images = data['images2_output'].to(self.opt.weight_dtype) # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output'].to(self.opt.weight_dtype) # [B, V, 1, output_size, output_size], ground-truth masks

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1).to(self.opt.weight_dtype) * (1 - gt_masks)

        loss_mse_image = F.mse_loss(pred_images.flatten(0,1), gt_images.flatten(0,1), reduction="none") 
        loss_mse_alpha = F.mse_loss(pred_alphas.flatten(0,1), gt_masks.flatten(0,1), reduction="none")
        loss_mse_image  = (loss_mse_image.mean(dim=list(range(1, len(loss_mse_image.shape)))) * mse_loss_weights).mean()
        loss_mse_alpha = (loss_mse_alpha.mean(dim=list(range(1, len(loss_mse_alpha.shape)))) * mse_loss_weights).mean()
        results['loss_mse_image'] = loss_mse_image
        results['loss_mse_alpha'] = loss_mse_alpha
        loss_mse = loss_mse_image + loss_mse_alpha
        results['loss_mse'] = loss_mse
        loss = loss + loss_mse
        
        results['gt_noise'] = noise_images.reshape(bsz // num_views, num_views, 3, 512, 512)

        if self.opt.lambda_lpips > 0 and step_ratio > 0:
            loss_lpips = self.lpips_loss(
                # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # downsampled to at most 256 to reduce memory cost
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            )
            lpips_loss_weights = torch.ones_like(mse_loss_weights)
            if start_idx is not None:
                lpips_loss_weights[start_idx::self.opt.num_views] = 5.0
            loss_lpips = (loss_lpips.mean(dim=list(range(1, len(loss_lpips.shape)))) * lpips_loss_weights).mean()
            results['loss_lpips'] = loss_lpips
            #loss = loss + self.opt.lambda_lpips * (step_ratio-0.25) * loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
            
        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr

        return results
