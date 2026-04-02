#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.train_report_utils import training_report
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


class Trainer:
    def __init__(self, args, dataset, opt, pipe, testing_iterations, saving_iterations) -> None:
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations

        self.tb_writer = prepare_output_and_logger(dataset)
        self.deform = DeformModel(deform_type=self.dataset.deform_type, is_blender=self.dataset.is_blender, 
                                  hyper_dim=self.dataset.hyper_dim,
                                  pred_color=self.dataset.pred_color)
        deform_loaded = self.deform.load_weights(dataset.model_path, iteration=None) #was -1
        self.deform.train_setting(opt)

        gs_fea_dim = self.dataset.hyper_dim
        self.gaussians = GaussianModel(dataset.sh_degree, no_binary_separation=self.dataset.no_binary_separation,
                                       fea_dim=gs_fea_dim)

        self.scene = Scene(dataset, self.gaussians, load_iteration=None) #originally was -1, we always want fresh start
        self.gaussians.training_setup(opt)
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1 if self.scene.loaded_iter is None else self.scene.loaded_iter

        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_ms_ssim = 0.0
        self.best_lpips = np.inf
        self.best_alex_lpips = np.inf
        self.best_iteration = 0
        self.progress_bar = tqdm.tqdm(range(opt.iterations), desc="Training progress")
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)       
        self.T_current = 0.5

    # no gui mode
    def train(self, iters=5000):
        if iters > 0:
            for i in tqdm.trange(iters):
                self.train_step()
    
    def train_step(self):
        self.iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree 
        if self.iteration % self.opt.oneupSHdegree_step == 0:
            self.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            viewpoint_stack = self.scene.getTrainCameras().copy()
            self.viewpoint_stack = viewpoint_stack
        
        time_interval = 1 / len(self.scene.all_timesteps)

        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        #when start binarization
        if self.iteration>self.opt.binarization_warm_up and not self.dataset.no_binary_separation:
            self.gaussians.no_binary_separation = False
        else:
            self.gaussians.no_binary_separation = True


        if self.deform.name == 'mlp' or self.deform.name == 'static':
            if self.iteration < self.opt.warm_up:
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                N = self.gaussians.get_xyz.shape[0]
                time_input = fid.unsqueeze(0).expand(N, -1)
                ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(self.iteration)
                d_values = self.deform.step(self.gaussians.get_xyz, time_input + ast_noise, iteration=self.iteration, 
                                            feature=self.gaussians.get_binary_feature(eval=False, T=self.T_current),
                                            camera_center=viewpoint_cam.camera_center)
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        else:
            raise NotImplemented
            
        # Render
        render_pkg_re = render(viewpoint_cam, self.gaussians, self.pipe, self.background, d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re["viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
    


        lambda_normal = 0.02 if self.iteration > self.opt.start_normal_reg else 0.0
        lambda_dist = self.opt.lambda_dist if self.iteration > self.opt.start_normal_reg else 0.0
        rend_dist = render_pkg_re["rend_dist"]
        rend_normal  = render_pkg_re['rend_normal']
        surf_normal = render_pkg_re['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

    
        
        # Loss
        gt_image = viewpoint_cam.original_image_train_light.cuda()
        if self.dataset.white_background and viewpoint_cam.gt_alpha_mask is not None and self.opt.gt_alpha_mask_as_scene_mask:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * self.background[:, None, None]

        Ll1 = l1_loss(image, gt_image)
        loss_img = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = loss_img + normal_loss + dist_loss

        #mask loss
        if self.opt.gt_alpha_mask_as_scene_mask and viewpoint_cam.gt_alpha_mask is not None:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            alpha_loss = F.binary_cross_entropy(render_pkg_re['rend_alpha'][:, None, None], gt_alpha_mask.unsqueeze(1).unsqueeze(1))
            loss += alpha_loss*self.opt.lambda_alpha_loss

        else:
            simulated_mask = torch.ones_like(render_pkg_re['rend_alpha'][:, None, None])
            alpha_loss = F.binary_cross_entropy(render_pkg_re['rend_alpha'][:, None, None], simulated_mask)
            loss += alpha_loss * 0.001


        if self.iteration > self.opt.warm_up:

            # d_xyz loss
            loss += ((d_xyz**2).mean())*self.opt.d_xyz_loss_weight 
            
            # d color loss
            d_color_reg_loss_weight = self.opt.d_color_reg_loss_weight
            if (d_color is not None and torch.is_tensor(d_color)):
                
                shadow_modulation = d_color[:, :3]

                d_color_reg_loss = (
                    shadow_modulation.pow(2.0).mean() * d_color_reg_loss_weight
                )
                
                loss += d_color_reg_loss

            if self.iteration > self.opt.binarization_warm_up and not self.dataset.no_binary_separation:

                # L1 for unsupervised bianrizationin the paper
                loss += (self.gaussians.get_binary_feature(eval=False, T=self.T_current)**1).mean()*self.opt.lambda_separation


        loss.backward()

        self.iter_end.record()

        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                self.progress_bar.update(10)
            if self.iteration == self.opt.iterations:
                self.progress_bar.close()

            # Keep track of max radii in image-space for pruning
            if self.gaussians.max_radii2D.shape[0] == 0:
                self.gaussians.max_radii2D = torch.zeros_like(radii)
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # Log and save
            cur_psnr, cur_ssim, cur_lpips, cur_ms_ssim, cur_alex_lpips = training_report(self.tb_writer, self.iteration, Ll1, 
                                                                                         loss, l1_loss, self.iter_start.elapsed_time(self.iter_end), 
                                                                                         self.testing_iterations, self.scene, render, 
                                                                                         (self.pipe, self.background), self.deform, 
                                                                                         self.dataset.load2gpu_on_the_fly, self.progress_bar)
            if self.iteration in self.testing_iterations:
                if cur_psnr.item() > self.best_psnr:
                    self.best_psnr = cur_psnr.item()
                    self.best_iteration = self.iteration
                    self.best_ssim = cur_ssim.item()
                    self.best_ms_ssim = cur_ms_ssim.item()
                    self.best_lpips = cur_lpips.item()
                    self.best_alex_lpips = cur_alex_lpips.item()

            if self.iteration in self.saving_iterations or self.iteration == self.best_iteration or self.iteration == self.opt.warm_up-1:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration)
                self.deform.save_weights(self.args.model_path, self.iteration)

            # Densification
            if self.iteration < self.opt.densify_until_iter:
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    print("Gaussian numberBEFORE PRUNE", len(self.gaussians.get_xyz))
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, self.opt.min_opacity, self.scene.cameras_extent, size_threshold)
                    print("Gaussian numberAFTER PRUNE", len(self.gaussians.get_xyz))

                if self.iteration % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            # Optimizer step
            if self.iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.update_learning_rate(self.iteration)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.deform.optimizer.step()
                self.deform.optimizer.zero_grad()
                self.deform.update_learning_rate(self.iteration)
                
        self.deform.update(max(0, self.iteration - self.opt.warm_up))

        self.progress_bar.set_description("Best PSNR={} in Iteration {}, SSIM={}, LPIPS={}, MS-SSIM={}, ALex-LPIPS={}".format('%.5f' % self.best_psnr, self.best_iteration, '%.5f' % self.best_ssim, '%.5f' % self.best_lpips, '%.5f' % self.best_ms_ssim, '%.5f' % self.best_alex_lpips))
        self.iteration += 1

   

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    

    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[1, 1000, 3000, 5000, 10000, 20000] + list(range(10000, 100_0001, 10000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 1000, 10000, 20000]+ list(range(10000, 100_0001, 10000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform-type", type=str, default='mlp')

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)


    if not args.model_path.endswith(args.deform_type):
        args.model_path = os.path.join(os.path.dirname(os.path.normpath(args.model_path)), os.path.basename(os.path.normpath(args.model_path)) + f'_{args.deform_type}')
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    trainer = Trainer(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args),testing_iterations=args.test_iterations, saving_iterations=args.save_iterations)


    trainer.train(args.iterations)
    
    # All done
    print("\nTraining complete.")
