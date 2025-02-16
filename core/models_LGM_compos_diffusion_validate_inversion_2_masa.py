import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import kiui
from kiui.lpips import LPIPS

from core.unet_LGM_compos import UNet
from core.options_latents_diffusion import Options
from core.gs import GaussianRenderer
from diffusers import AutoencoderKL, DDPMScheduler,  UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional
import random
import torchvision.transforms.functional as TF
import tqdm
from core.control import ControlNetPipeline
from core.masactrl import MutualSelfAttention3DControl
from core.masactrl_utils import AttentionBase, regiter_attention_editor_diffusers

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
        ).to(self.opt.weight_dtype)

        # last conv
        self.conv = nn.Conv2d(14, 14, kernel_size=1).to(self.opt.weight_dtype) # NOTE: maybe remove it if train again

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

        self.unet2 = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", low_cpu_mem_usage=False,device_map=None,ignore_mismatched_sizes=True).to(self.opt.weight_dtype)
        self.unet3 = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", low_cpu_mem_usage=False,device_map=None,ignore_mismatched_sizes=True).to(self.opt.weight_dtype)
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.opt.weight_dtype)
        self.tokenizer =  CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler2 = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.test_scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

        
        #self.pipe = MasaCtrlPipeline.from_pretrained(model_key, scheduler=self.test_scheduler)
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.opt.weight_dtype)
        self.vae.requires_grad_(False)
        self.unet2.requires_grad_(False)
        self.unet3.requires_grad_(False)
        #self.tokenizer.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.steps = 2
        self.layer = 10
        #self.masa_editor = MutualSelfAttention3DControl(step, layer, total_steps=30)
        self.base_editor = AttentionBase()

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

    
    def prepare_default_rays(self, device, elevation=0, proj_matrix=None):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius, opengl=True),
            orbit_camera(elevation, 90, radius=self.opt.cam_radius, opengl=True),
            orbit_camera(elevation, 180, radius=self.opt.cam_radius, opengl=True),
            orbit_camera(elevation, 270, radius=self.opt.cam_radius, opengl=True),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_ray_size, self.opt.input_ray_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure
            # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        cam_poses[:, :3, 1:3] *= -1
        cam_poses = cam_poses.to(device)
        cam_view = torch.inverse(cam_poses).transpose(1, 2)
        cam_view_proj = cam_view @ proj_matrix
        cam_pos = - cam_poses[:, :3, 3]

        return rays_embeddings, cam_view, cam_view_proj, cam_pos
        
    
    def prepare_default_rays_zero123(self, device, elevation=0, proj_matrix=None):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(0, 0, radius=self.opt.cam_radius, opengl=True),
            orbit_camera(-10, 90, radius=self.opt.cam_radius, opengl=True),
            orbit_camera(-10, 210, radius=self.opt.cam_radius, opengl=True),
            orbit_camera(20, 270, radius=self.opt.cam_radius, opengl=True),
        ], axis=0) # [4, 4, 4]
        # cam_poses = np.stack([
        #     orbit_camera(0,  0, radius=self.opt.cam_radius, opengl=True),
        #     orbit_camera(0, 120, radius=self.opt.cam_radius, opengl=True),
        #     orbit_camera(0, 240, radius=self.opt.cam_radius, opengl=True),
        #     orbit_camera(-30, 300, radius=self.opt.cam_radius, opengl=True),
        # ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_ray_size, self.opt.input_ray_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure
            # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        cam_poses[:, :3, 1:3] *= -1
        cam_poses = cam_poses.to(device)
        cam_view = torch.inverse(cam_poses).transpose(1, 2)
        cam_view_proj = cam_view @ proj_matrix
        cam_pos = - cam_poses[:, :3, 3]

        return rays_embeddings, cam_view, cam_view_proj, cam_pos
    
    def unet_step(
        self,
        model_output: torch.FloatTensor,
        timestep,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
        """
        prev_timestep = timestep - self.test_scheduler.config.num_train_timesteps // self.test_scheduler.num_inference_steps
        alpha_prod_t = self.test_scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.test_scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.test_scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev
    
    def forward_gaussians(self, images, encoder_hidden_states, data, uncon_encoder_hidden_states=None):
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)
        timestep = data["timesteps"].flatten(0, 1)
        pred_noise, blocks_sample, temb= self.unet2(images, timestep, encoder_hidden_states, return_dict=False)
        if uncon_encoder_hidden_states is not None:
            uncon_pred_noise, _, _= self.unet3(images, timestep, uncon_encoder_hidden_states, return_dict=False)
            pred_noise = uncon_pred_noise + 3 * (pred_noise - uncon_pred_noise)
            # print(3.5)
        if pred_noise.shape[0] == 5:
            pred_x0 = self.pred_x0(pred_noise[:4], timestep[:4], images[:4])
            masa_latent = self.unet_step(pred_noise[4:,], timestep[4].item(), images[4:])
            temb = temb[:4]
            blocks_sample = [i[:4] for i in blocks_sample]
        else:
            pred_x0 = self.pred_x0(pred_noise, timestep, images)
        images_512 = (self.vae.decode(pred_x0.to(self.opt.weight_dtype) / 0.18215).sample +1)*0.5
        images_256 = F.interpolate(images_512.clamp(0, 1), (256, 256), mode='bilinear', align_corners=False)
        images_256 = TF.normalize(images_256, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        images_256 = torch.cat([images_256.to(self.opt.weight_dtype), data['ray'].to(self.opt.weight_dtype) ], dim=1)

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
        
        return gaussians, images_512, masa_latent
    
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
        alphas_cumprod = self.test_scheduler.alphas_cumprod.to(device=x.device)
        alpha_prod_t = alphas_cumprod [timestep]

        B = alpha_prod_t.shape[0]
        alpha_prod_t = alpha_prod_t.view(B, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        return pred_x0
    
    def step(
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
        prev_timestep = timestep - self.test_scheduler.config.num_train_timesteps // self.test_scheduler.num_inference_steps
        prev_timestep[timestep==0] = 0
        alphas_cumprod = self.test_scheduler.alphas_cumprod.to(device=x.device)
        alpha_prod_t = alphas_cumprod [timestep]
        #alpha_prod_t_prev = self.test_scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.test_scheduler.final_alpha_cumprod
        alpha_prod_t_prev = torch.where(prev_timestep >0, self.test_scheduler.alphas_cumprod[prev_timestep], self.test_scheduler.final_alpha_cumprod).to(device=x.device)
        B = alpha_prod_t.shape[0]
        alpha_prod_t = alpha_prod_t.view(B, 1, 1, 1)
        alpha_prod_t_prev = alpha_prod_t_prev.view(B, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        
        #pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_noise = (x - alpha_prod_t**0.5 * model_output) / beta_prod_t**0.5
        
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * pred_noise
        #x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        x_prev = alpha_prod_t_prev**0.5 * model_output + pred_dir
        return x_prev
    
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
            timesteps[::num_views] = 0
            timesteps_pred[::self.opt.num_views] = 0
            
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
            if timesteps[0] == 0:
                lpips_loss_weights[::self.opt.num_views] = 5.0
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
    
    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.test_scheduler.config.num_train_timesteps // self.test_scheduler.num_inference_steps, 999)
        alpha_prod_t = self.test_scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.test_scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.test_scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0
    
    # def invert(self, image, encoder_hidden_states):
    #     noisy_latent = image
    #     for i, t in enumerate(reversed(self.test_scheduler.timesteps)):
    #         model_inputs = noisy_latent
    #         noise_pred = self.unet2(model_inputs, t, encoder_hidden_states=encoder_hidden_states).sample
    #         noisy_latent, pred_x0 = self.next_step(noise_pred, t, noisy_latent)
    #         a = (self.vae.decode(pred_x0.detach()/ 0.18215).sample +1)*0.5
    #         b = a.clamp(0,1).float().reshape(8, 4, 3, 512, 512).detach().to(torch.float).cpu().numpy()
    #         c1 = b.transpose(0, 3, 1, 4, 2).reshape(-1, b.shape[1] * b.shape[3], 3)
    #         kiui.write_image(f'{i}_2.jpg', c1)
    #     return noisy_latent
    
    @torch.no_grad()
    def image2latent(self, image):
        #DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # if type(image) is Image:
        #     image = np.array(image)
        #     image = torch.from_numpy(image).float() / 127.5 - 1
        #     image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents
    
    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt="",
        # num_inference_steps=50,
        # guidance_scale=7.5,
        # eta=0.0,
        # return_intermediates=False,
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = image.device
        batch_size = image.shape[0]
        # if isinstance(prompt, list):
        #     if batch_size == 1:
        #         image = image.expand(len(prompt), -1, -1, -1)
        # elif isinstance(prompt, str):
        #     if batch_size > 1:
        #         prompt = [prompt] * batch_size
        prompt = [prompt] * batch_size
        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents
        # print(latents)
        # exit()
        # unconditional embedding for classifier free guidance
        # if guidance_scale > 1.:
        #     max_length = text_input.input_ids.shape[-1]
        #     unconditional_input = self.tokenizer(
        #         [""] * batch_size,
        #         padding="max_length",
        #         max_length=77,
        #         return_tensors="pt"
        #     )
        #     unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
        #     text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        # print("latents shape: ", latents.shape)
        # interative sampling
        #self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.test_scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(reversed(self.test_scheduler.timesteps)):
            # if guidance_scale > 1.:
            #     model_inputs = torch.cat([latents] * 2)
            model_inputs = latents

            # predict the noise
            noise_pred = self.unet3(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            # if guidance_scale > 1.:
            #     noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
            #     noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            # a = (self.vae.decode(pred_x0.detach()/ 0.18215).sample +1)*0.5
            # b = a.clamp(0,1).float().reshape(8, 4, 3, 512, 512).detach().to(torch.float).cpu().numpy()
            # c1 = b.transpose(0, 3, 1, 4, 2).reshape(-1, b.shape[1] * b.shape[3], 3)

            a = (self.vae.decode(pred_x0[:1].detach()/ 0.18215).sample +1)*0.5
            b = a.clamp(0,1).float().reshape(1, 3, 512, 512).detach().to(torch.float).cpu().numpy()
            c1 = b.transpose(0, 2, 3, 1)
            kiui.write_image(f'{self.opt.workspace}/{i}_7.jpg', c1)

            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        # if return_intermediates:
        #     # return the intermediate laters during inversion
        #     # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
        #     return latents, latents_list
        return latents
    
    def validate(self, data, num_inference_steps=30, single_image=True):
        results = {}
        self.test_scheduler.set_timesteps(num_inference_steps)
        self.opt.weight_dtype = torch.bfloat16
        data['input'] =  self.vae.encode(data['images2_output']*2 -1).latent_dist.mode().detach() *0.18215
        data['input'] = data['input'].unsqueeze(0)
        images = data['input'].to(self.opt.weight_dtype) # [B, 4, 9, h, W], input features
        
        self.masa_editor = MutualSelfAttention3DControl(self.steps, self.layer, total_steps=num_inference_steps)
        self.masa_editor.reset()
        regiter_attention_editor_diffusers(self.unet2, self.masa_editor)
        #self.test_scheduler = self.test_scheduler.to(images.device)

        num_views = images.shape[1]
        #ray_embedding = images[:, :, 4:]
        latents = images.flatten(0,1)
        latent = latents[:,:4]
        bsz, c, h, w = latent.shape
        
        gt_images = data['images2_output'].to(self.opt.weight_dtype)
        
        noise = torch.randn_like(latent).to(device=images.device)
        data['noisy_latents'] = noise.reshape(bsz // num_views, num_views, c, h, w).to(self.opt.weight_dtype)
        
        prompt = ['']*images.shape[0]
        uncon_encoder_hidden_states = self.encode_prompt(prompt, images.device).to(images.dtype)
        uncon_encoder_hidden_states = uncon_encoder_hidden_states[:,None].repeat(1,images.shape[1], 1, 1)
        uncon_encoder_hidden_states = uncon_encoder_hidden_states.flatten(0,1)
        
        prompt = data["prompt"]*(images.shape[0])
            
        encoder_hidden_states = self.encode_prompt(prompt, images.device).to(images.dtype)
        encoder_hidden_states = encoder_hidden_states[:,None].repeat(1,images.shape[1], 1, 1)
        encoder_hidden_states = encoder_hidden_states.flatten(0,1)
        encoder_hidden_states[4:] = uncon_encoder_hidden_states[4:]

        img_latent =  self.vae.encode(gt_images*2 -1).latent_dist.mode().detach() *0.18215
        img = (self.vae.decode(img_latent.to(self.opt.weight_dtype) / 0.18215).sample +1)*0.5
        #img = gt_images
        data['noisy_latents'] = self.invert(img*2-1).reshape(bsz // num_views, num_views, c, h, w)

        # timesteps
        # timesteps = torch.ones((bsz // num_views,), device=images.device)* 481
        # timesteps_pred = timesteps.repeat_interleave(self.opt.num_views)
        # timesteps = timesteps.repeat_interleave(num_views)
        # timesteps = timesteps.long()
        # # timesteps[::num_views] = 0
        # # timesteps_pred[::self.opt.num_views] = 0
        # # add noise
        # noise = torch.randn_like(latent).to(device=images.device)
        # noisy_latents = self.test_scheduler.add_noise(latent, noise, timesteps).to(device=images.device)
        # data['noisy_latents'] = noisy_latents.reshape(bsz // num_views, num_views, c, h, w)
        # data['timesteps'] = timesteps.reshape(bsz // num_views, num_views)
        
        if single_image is True:
            data['noisy_latents'][:, :1] = images[:, :1]
        for i, t in enumerate(self.test_scheduler.timesteps):
            
            print(i, t)
            # timesteps = torch.ones((bsz // num_views,), device=images.device)* t
            # timesteps_pred = timesteps.repeat_interleave(self.opt.num_views)
            # timesteps = timesteps.repeat_interleave(num_views)
            # timesteps = timesteps.long()
            # # timesteps[::num_views] = 0
            # # timesteps_pred[::self.opt.num_views] = 0
            # # add noise
            # noise = torch.randn_like(latent).to(device=images.device)
            # noisy_latents = self.test_scheduler.add_noise(latent, noise, timesteps).to(device=images.device)
            # data['noisy_latents'] = noisy_latents.reshape(bsz // num_views, num_views, c, h, w)
            
            timesteps = t.repeat(bsz // num_views)
            #timesteps_pred = timesteps.repeat_interleave(self.opt.num_views)
            timesteps = timesteps.repeat_interleave(num_views)
            timesteps = timesteps.long()
            #if(random.random() < 0.9):
            if single_image is True:
                timesteps[::num_views] = 0
                #timesteps_pred[::self.opt.num_views] = 0

            # add noise
            # noise = torch.randn_like(latent).to(device=images.device)
            # noisy_latents = self.scheduler.add_noise(latent, noise, timesteps).to(device=images.device)
            # data['noisy_latents'] = noisy_latents.reshape(bsz // num_views, num_views, c, h, w)
            data['timesteps'] = timesteps.reshape(bsz // num_views, num_views).to(device=images.device)
            timesteps_cpu = timesteps.reshape(bsz // num_views, num_views)
            ### FIXME
            #timesteps_pred = torch.cat([data["timesteps"], 300 * torch.ones(self.opt.batch_size, self.opt.num_views-data['timesteps'].shape[1]).long().to(timesteps.device)],dim=1).flatten(0,1)
            #snr = self.compute_snr(timesteps_pred)
            #mse_loss_weights = torch.stack([snr, opt.snr_gamma * torch.ones_like(timesteps_pred)], dim=1).min(dim=1)[0] / snr
            #mse_loss_weights = torch.stack([snr, self.opt.snr_gamma * torch.ones_like(timesteps_pred)], dim=1).min(dim=1)[0] 
            # use the first view to predict gaussians
            # prompt = ['']*images.shape[0]
            # uncon_encoder_hidden_states = self.encode_prompt(prompt, images.device).to(images.dtype)
            # uncon_encoder_hidden_states = uncon_encoder_hidden_states[:,None].repeat(1,images.shape[1], 1, 1)
            # uncon_encoder_hidden_states = uncon_encoder_hidden_states.flatten(0,1)
            # uncon_encoder_hidden_states = None
             
            
            images = data['noisy_latents']
            # img = (self.vae.decode(images.flatten(0,1).to(self.opt.weight_dtype) / 0.18215).sample +1)*0.5
            # b = img.unsqueeze(0).clamp(0,1).detach().to(torch.float).cpu().numpy() 
            # c1 = b.transpose(0, 3, 1, 4, 2).reshape(-1, b.shape[1] * b.shape[3], 3)
            # kiui.write_image(f'{self.opt.workspace}/{i}_1_noise.jpg', c1)
            gaussians, noise_images, masa_latent = self.forward_gaussians(images, encoder_hidden_states, data, uncon_encoder_hidden_states) # [B, N, 14]

            results['gaussians'] = gaussians

            bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
            # use the other views for rendering and supervision
            results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color, scale_modifier=1)
            # pred_images = results['image'] # [B, V, C, output_size, output_size]
            pred_alphas = results['alpha'].to(self.opt.weight_dtype)
            pred_images = results['image'].to(self.opt.weight_dtype)
            #pred_images = pred_images + self.white_latent.to(pred_images.device)*(1-pred_alphas)
            
            
            #data['noisy_latents'] = self.step((self.vae.encode(pred_images[:,:4].flatten(0, 1)*2 -1).latent_dist.mode().detach())*0.18215, timesteps_cpu.flatten(0, 1), data['noisy_latents'].flatten(0, 1)).reshape(bsz // num_views, num_views, c, h, w).to(self.opt.weight_dtype)
            data['noisy_latents'] = torch.cat([self.step((self.vae.encode(pred_images[:,:4].flatten(0, 1)*2 -1).latent_dist.mode().detach())*0.18215, timesteps_cpu[:,:4].flatten(0, 1), data['noisy_latents'][:,:4].flatten(0, 1)), masa_latent]).reshape(bsz // num_views, num_views, c, h, w).to(self.opt.weight_dtype)
            # if t > 0:
            #     #data['noisy_latents'] = self.step((self.vae.encode(pred_images[:,:4].flatten(0, 1)*2 -1).latent_dist.mode().detach())*0.18215, timesteps_cpu.flatten(0, 1), noise).reshape(bsz // num_views, num_views, c, h, w).to(self.opt.weight_dtype)
            #     data['noisy_latents'] = self.scheduler.add_noise((self.vae.encode(pred_images[:,:4].flatten(0, 1)*2 -1).latent_dist.mode().detach())*0.18215, noise, timesteps-1).reshape(bsz // num_views, num_views, c, h, w).to(self.opt.weight_dtype)

            if single_image is True:
                data['noisy_latents'][:, :1] = images[:, :1]
            
            #a = (self.vae.decode(pred_images.detach().to(dtype=torch.bfloat16).flatten(0,1)/ 0.18215).sample +1)*0.5
            b = pred_images.detach().to(torch.float).cpu().numpy() 
            c1 = b.transpose(0, 3, 1, 4, 2).reshape(-1, b.shape[1] * b.shape[3], 3)
            kiui.write_image(f'{self.opt.workspace}/{i}_2.jpg', c1)
            
            #a = (self.vae.decode(data['noisy_latents'].detach().flatten(0,1)/ 0.18215).sample +1)*0.5
            b = noise_images.clamp(0,1).float().reshape(1, 4, 3, 512, 512).detach().to(torch.float).cpu().numpy()
            c1 = b.transpose(0, 3, 1, 4, 2).reshape(-1, b.shape[1] * b.shape[3], 3)
            kiui.write_image(f'{self.opt.workspace}/{i}_2_noise.jpg', c1)
            

        return results, gaussians
    