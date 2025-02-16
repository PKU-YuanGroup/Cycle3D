import tyro
import time
import random

import torch
from core.options_latents_diffusion import AllConfigs
from core.models_LGM_compos_diffusion import LGM
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file
from torch.utils.tensorboard import SummaryWriter
import kiui
from diffusers.utils.import_utils import is_xformers_available
import os
import shutil


def main():    
    opt = tyro.cli(AllConfigs)
    
    writer = SummaryWriter(f'{opt.workspace}/runs')
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    print(opt.pretrained_model_name_or_path)
    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        # kwargs_handlers=[ddp_kwargs],
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        opt.mixed_precision = accelerator.mixed_precision
        opt.weight_dtype = weight_dtype
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        opt.weight_dtype = weight_dtype
        opt.mixed_precision = accelerator.mixed_precision

    # model
    model = LGM(opt)
    # vae = model.vae
    # text_encoder = model.text_encoder
    # text_encoder.requires_grad_(False)
    # vae.requires_grad_(False)
    
    unet = model.unet
    conv = model.conv
    unet.requires_grad_(True)
    conv.requires_grad_(True)
    unet2 = model.unet2

    if opt.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            unet2.enable_xformers_memory_efficient_attention()
    
            
    if opt.gradient_checkpointing:
        unet2.enable_gradient_checkpointing()
    
    # resume
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        
        # tolerant load (only load matching shapes)
        # model.load_state_dict(ckpt, strict=False)
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')
    
    # data
    if opt.data_mode == 's3':
        from core.provider_Gobjaverse_latent_diffusion_insert import GobjaverseDataset as Dataset
    else:
        raise NotImplementedError

    train_dataset = Dataset(opt, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = Dataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    # if opt.gradient_checkpointing:
    #     model.enable_gradient_checkpointing()
    
    #params = []
    # for name, param in unet.named_parameters():
    #     #if name.startswith(tuple(('up_blocks', 'mid_block', 'conv_out'))):
    #     params.append(param) 
    # for name, param in conv.named_parameters():
    #     params.append(param) 
    params = []
    for name, param in model.named_parameters():
        if name.startswith('unet.'):
            #print(name)
            params.append(param) 
        elif not name.startswith(tuple(('unet2', 'vae', 'tokenizer', 'text_encoder', 'scheduler', 'lpips'))):
            #print(name)
            params.append(param)

    # optimizer
    optimizer = torch.optim.AdamW(params, lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))

    # scheduler (per-iteration)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, eta_min=1e-6)
    total_steps = opt.num_epochs * len(train_dataloader)
    pct_start = 3000 / total_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    # loop
    for epoch in range(opt.num_epochs):
        # train
        model.train()
        total_loss = 0
        total_psnr = 0

        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                optimizer.zero_grad()

                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                out = model(data, step_ratio)
                loss = out['loss']
                psnr = out['psnr']
                accelerator.backward(loss)
                
                writer.add_scalar('loss', loss.item(), epoch*len(train_dataloader)+i)
                #writer.add_scalar('loss_mse', out['loss_mse'].item(), epoch*len(train_dataloader)+i)
                writer.add_scalar('loss_mse_image', out['loss_mse_image'].item(), epoch*len(train_dataloader)+i)
                writer.add_scalar('loss_mse_alpha', out['loss_mse_alpha'].item(), epoch*len(train_dataloader)+i)
                if step_ratio> 0:
                    writer.add_scalar('loss_lpips', out['loss_lpips'].item(), epoch*len(train_dataloader)+i)
                writer.add_scalar('psnr', psnr.item(), epoch*len(train_dataloader)+i)
                writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch*len(train_dataloader)+i)
                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                scheduler.step()

                total_loss += loss.detach()
                total_psnr += psnr.detach()

            if accelerator.is_main_process:
                # logging
                if i % 100 == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()    
                    print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} step_ratio: {step_ratio:.4f} loss: {loss.item():.6f} loss_mse: {out['loss_mse_image']:.6f}")

                # save log images
                if i % 200 == 0:
                    ## FIXME
                    ## 3 ------>4 
                    with torch.no_grad():
                        # gt_images = (vae.decode(data['images_output'][0, :8].detach().to(dtype=torch.bfloat16)/ 0.18215).sample +1)*0.5
                        # gt_images = gt_images.clamp(0,1).float().unsqueeze(0).detach().cpu().numpy() 
                        # #gt_images = data['images_output'][:1].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        # gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                        # kiui.write_image(f'{opt.workspace}/train_gt_images_{epoch}_{i}.jpg', gt_images)

                        gt_alphas = data['masks_output'].clamp(0,1).float().detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                        gt_alphas = gt_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, gt_alphas.shape[1] * gt_alphas.shape[3], 1)
                        kiui.write_image(f'{opt.workspace}/train_gt_alphas_{epoch}_{i}.jpg', gt_alphas)
                        
                        # gt_images_ori = (vae.decode((data['images_output'].detach()*data['masks_output']+out['white_latent'].detach()*(1-data['masks_output']))[0, :8].to(dtype=torch.bfloat16)/ 0.18215).sample +1)*0.5
                        # gt_images_ori = gt_images_ori.clamp(0,1).float().unsqueeze(0).detach().cpu().numpy() 
                        # gt_images_ori = gt_images_ori.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images_ori.shape[1] * gt_images_ori.shape[3], 3) # [B*output_size, V*output_size, 3]
                        # kiui.write_image(f'{opt.workspace}/train_gt_images_ori_{epoch}_{i}.jpg', gt_images_ori)
                        
                        gt_noise_images = out["gt_noise"].clamp(0,1).float().detach().cpu().numpy()
                        #gt_noise_images = gt_noise_images.transpose(0, 2, 3, 1).reshape(-1, gt_noise_images.shape[2], 3)
                        gt_noise_images = gt_noise_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_noise_images.shape[1] * gt_noise_images.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/train_gt_noise_images_{epoch}_{i}.jpg', gt_noise_images)

                        gt_images = data['images2_output'].clamp(0,1).float().detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    
                        # data['images_output'] = (vae.decode(data['images_output'][0, :4].to(dtype=torch.bfloat16)/ 0.18215).sample +1)*0.5
                        # gt_images = data['images_output'].clamp(0,1).float().unsqueeze(0).detach().cpu().numpy() 
                        #gt_images = data['images_output'][:1].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        # gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                        kiui.write_image(f'{opt.workspace}/train_gt_images_{epoch}_{i}.jpg', gt_images)
                        
                        # out['images_pred'] = (vae.decode(out['images_pred'][0, :8].detach().to(dtype=torch.bfloat16)/ 0.18215).sample +1)*0.5
                        pred_images = out['images_pred'].clamp(0,1).float().detach().cpu().numpy() 
                        #pred_images = out['images_pred'].reshape(data['images_output'].shape[0],data['images_output'].shape[1], *out['images_pred'].shape[1:]).detach().cpu().numpy() 

                        #pred_images = out['images_pred'][:1].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/train_pred_images_{epoch}_{i}.jpg', pred_images)
  
                        # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                        # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                        # kiui.write_image(f'{opt.workspace}/train_pred_alphas_{epoch}_{i}.jpg', pred_alphas)

        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")
            
        # checkpoint
        if epoch % 1 == 0 or epoch == opt.num_epochs - 1:
            accelerator.wait_for_everyone()
            accelerator.save_model(model, opt.workspace)
        
        
        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
        # if opt.checkpoints_total_limit is not None:

        #     checkpoints = os.listdir(opt.workspace)
        #     checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        #     checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        #     # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        #     if len(checkpoints) >= opt.checkpoints_total_limit:
        #         num_to_remove = len(checkpoints) - opt.checkpoints_total_limit + 1
        #         removing_checkpoints = checkpoints[0:num_to_remove]

        #         print(
        #             f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
        #         )
        #         print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

        #         for removing_checkpoint in removing_checkpoints:
        #             removing_checkpoint = os.path.join(opt.workspace, removing_checkpoint)
        #             shutil.rmtree(removing_checkpoint)

        # save_path = os.path.join(opt.workspace, f"checkpoint-{epoch}")
        # accelerator.save_state(save_path)
        #print(f"Saved state to {save_path}")
        
        # eval
        with torch.no_grad():
            model.eval()
            total_psnr = 0
            for i, data in enumerate(test_dataloader):

                out = model(data)
    
                psnr = out['psnr']
                total_psnr += psnr.detach()
                
                # save some images
                if accelerator.is_main_process:
                    gt_images = data['images2_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/eval_gt_images_{epoch}_{i}.jpg', gt_images)

                    pred_images = out['images_pred'].clamp(0,1).float().detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/eval_pred_images_{epoch}_{i}.jpg', pred_images)

                    # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                    # kiui.write_image(f'{opt.workspace}/eval_pred_alphas_{epoch}_{i}.jpg', pred_alphas)

            torch.cuda.empty_cache()

            total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
            if accelerator.is_main_process:
                total_psnr /= len(test_dataloader)
                accelerator.print(f"[eval] epoch: {epoch} psnr: {psnr:.4f}")
    
    writer.close()

if __name__ == "__main__":
    main()
