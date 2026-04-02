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

from scene.light import EnvLight
from utils.loss_utils import l1_loss, ssim, first_order_edge_aware_loss, tv_loss
from gaussian_renderer.render_ir import render_ir
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state
import uuid
import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.train_report_utils import training_report_relight_screen_space
import torch.nn.functional as F


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


class Trainer:
    def __init__(self, args, dataset, opt, pipe, testing_iterations, saving_iterations, load_iter) -> None:

        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations
        self.n_samples = pipe.diffuse_sample_num

        self.tb_writer = prepare_output_and_logger(dataset)
        self.deform = DeformModel(deform_type=self.dataset.deform_type,
                                  is_blender=self.dataset.is_blender,
                                  hyper_dim=self.dataset.hyper_dim,
                                  pred_color=self.dataset.pred_color)
        deform_loaded = self.deform.load_weights(dataset.model_path, iteration=load_iter)
        self.deform.train_setting(opt)

        gs_fea_dim = self.dataset.hyper_dim
        self.gaussians = GaussianModel(dataset.sh_degree, no_binary_separation=dataset.no_binary_separation,
                                       fea_dim=gs_fea_dim)

        self.scene = Scene(dataset, self.gaussians, load_iteration=load_iter)

        # lower lr for most of gaussian params
        lr_scale = 0.1
        opt.position_lr_init *= lr_scale
        opt.opacity_lr *= lr_scale
        opt.scaling_lr *= lr_scale
        opt.rotation_lr *= lr_scale
        opt.albedo_after_stage1_lr *= lr_scale
        opt.albedo_rest_lr *= lr_scale
        opt.feature_lr *= lr_scale

        self.gaussians.training_setup(opt)

        # optimizer - only train colors and roughness and opacity
        remaining_params = [group for group in self.gaussians.optimizer.param_groups if
                            group["name"] in ["albedo_dc", "albedo_dc_stage1", "albedo_rest", "roughness", "opacity"]]
        self.gaussians.optimizer.param_groups.clear()
        self.gaussians.optimizer.param_groups.extend(remaining_params)
        self.gaussians.optimizer.state.clear()


        # optimize also mlp deform, but keep only color-related head
        before = sum(p.numel() for g in self.deform.optimizer.param_groups for p in g["params"])

        remaining_params = [group for group in self.deform.optimizer.param_groups if group["name"] in ["mlp_color"]]
        self.deform.optimizer.param_groups.clear()
        self.deform.optimizer.param_groups.extend(remaining_params)
        self.deform.optimizer.state.clear()
        for param_group in self.deform.optimizer.param_groups:
            param_group['lr'] = 0.000075
        
        after = sum(p.numel() for g in self.deform.optimizer.param_groups for p in g["params"])
        print(f"######## Optimizer params: before={before:,} | after={after:,}")

        self.original_train_cameras = self.scene.getTrainCameras()

        self.N = self.gaussians.get_xyz.shape[0]

        self.env_light = EnvLight(path=None, device='cuda',
                                  resolution=[opt.envmap_resolution // 2, opt.envmap_resolution],
                                  max_res=opt.envmap_resolution, init_value=opt.envmap_init_value,
                                  activation=opt.envmap_activation)
        self.env_light.training_setup(opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1 if self.scene.loaded_iter is None else self.scene.loaded_iter + 1

        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        self.progress_bar = tqdm.tqdm(range(opt.iterations), desc="Training progress")

        self.built_bvh = False
        self.edge_image_prepaired = False
        self.load_iter = load_iter

        self.iters_only_diffuse = 2000
        self.iters_only_diffuse_counter = 0


    def train(self, iters=5000):
        if (iters - self.scene.loaded_iter) > 0:
            for i in tqdm.trange(self.scene.loaded_iter, iters):
                self.train_step()

    def train_step(self):
        self.iter_start.record()

        # Pick a random Camera
        if not self.viewpoint_stack:
            viewpoint_stack = self.original_train_cameras.copy()
            self.viewpoint_stack = viewpoint_stack

        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))

        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        time_input = fid.unsqueeze(0).expand(self.N, -1)

        # careful with no grad
        with torch.no_grad():
            d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input, iteration=self.iteration,
                                        feature=self.gaussians.get_binary_feature(),
                                        camera_center=viewpoint_cam.camera_center)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], \
                                                           d_values['d_scaling'], d_values['d_opacity'], d_values[
                                                               'd_color']

        if not self.built_bvh:
            self.gaussians.build_bvh(d_rotation=d_rotation, d_xyz=d_xyz, d_scaling=d_scaling)
            self.built_bvh = True
        else:
            self.gaussians.update_bvh(d_rotation=d_rotation, d_xyz=d_xyz, d_scaling=d_scaling)

        # Render relight in screen space
        self.opt.train_ray = True
        render_pkg_re = render_ir(viewpoint_camera=viewpoint_cam,
                                  pc=self.gaussians, pipe=self.pipe,
                                  bg_color=self.background, d_xyz=d_xyz,
                                  d_rotation=d_rotation, d_scaling=d_scaling,
                                  d_opacity=d_opacity, d_color=d_color,
                                  relight=False, env_light=self.env_light, training=True, opt=self.opt)

        self.iters_only_diffuse_counter += 1
        if self.iters_only_diffuse_counter > self.iters_only_diffuse:
            image_non_masked = render_pkg_re["render"]
        else:
            image_non_masked = render_pkg_re["diffuse"]


        gt_image = viewpoint_cam.original_image_train_light.cuda()
        if self.opt.train_ray:
            mask = render_pkg_re["mask"]
            ray_rgb_gt = gt_image.permute(1, 2, 0)[mask]
            ray_rgb = image_non_masked.permute(1, 2, 0)[mask]
            Ll1 = F.l1_loss(ray_rgb, ray_rgb_gt)
        loss = Ll1

        ## add loss from stage1 colors
        rendered_image_sh = render_pkg_re["render_sh"]
        rend_alpha = render_pkg_re['rend_alpha']
        mask2 = (rend_alpha > 0.9).float()  # (B,1,H,W), we need mask for enerf scenes, where we have areas with manually removed gaussians

        if rendered_image_sh.shape[1] > 1:  # e.g. RGB
            mask2 = mask2.expand_as(rendered_image_sh)

        masked_render = rendered_image_sh * mask2
        masked_gt = gt_image * mask2

        lambda_dssim = 0.2
        loss_sh = (1.0 - lambda_dssim) * l1_loss(masked_render, masked_gt) \
                + lambda_dssim * (1.0 - ssim(masked_render, masked_gt))

        loss += loss_sh

        ### envmap loss
        if self.opt.d_lower_hemisphere_weight >0:
            env_dict = self.env_light.render_env_map(H=64)

            grid = [
                env_dict["env1"].permute(2, 0, 1),
                env_dict["env2"].permute(2, 0, 1),
            ]
            hdr_tensor = grid[1]  # C, H, W

            c, h, w = hdr_tensor.shape  # Get height, width, channels    
            penalty_h = round(h * 0.66)
            penalty = (hdr_tensor[:,penalty_h:,:]**2)

            loss_env_lowerhem = penalty.mean() 
            loss += self.opt.d_lower_hemisphere_weight*loss_env_lowerhem
        #########

        ##IRGS losses for tests:
        if self.opt.lambda_roughness_smooth > 0:
            rendered_roughness = render_pkg_re["roughness"]
            loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness * mask2, masked_gt)
            loss = loss + self.opt.lambda_roughness_smooth * loss_roughness_smooth
        
        if self.opt.lambda_light > 0:
            light_direct = render_pkg_re["ray_light_direct"]
            mean_light = light_direct.mean(-1, keepdim=True).expand_as(light_direct)
            loss_light = F.l1_loss(light_direct, mean_light)
            loss = loss + self.opt.lambda_light * loss_light

        if self.opt.lambda_light_smooth > 0:
            env = render_pkg_re["env_only"]
            loss_light_smooth = tv_loss(env)
            loss = loss + self.opt.lambda_light_smooth * loss_light_smooth
        
        if self.opt.lambda_base_color_smooth > 0:
            rendered_base_color = render_pkg_re["base_color_linear"]
            loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color*mask2, gt_image*mask2)
            loss = loss + self.opt.lambda_base_color_smooth * loss_base_color_smooth
        ###
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
                        
            training_report_relight_screen_space(self.tb_writer, self.iteration, Ll1, loss,
                                                 self.iter_start.elapsed_time(self.iter_end),
                                                 self.testing_iterations, self.scene,
                                                 (self.pipe, self.background), self.deform,
                                                 self.dataset.load2gpu_on_the_fly,
                                                 env_light=self.env_light)

            if self.iteration in self.saving_iterations or self.iteration == self.opt.warm_up - 1:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration)
                self.deform.save_weights(self.args.model_path, self.iteration)
                self.env_light.save_weights(self.args.model_path, self.iteration)

            # Optimizer step
            if self.iteration < self.opt.iterations and self.iteration % 4 == 0:
                self.gaussians.optimizer.step()
                self.gaussians.update_learning_rate(self.iteration)
                self.gaussians.optimizer.zero_grad(set_to_none=True)

                self.env_light.optimizer.step()
                self.env_light.optimizer.zero_grad(set_to_none=True)

                self.deform.optimizer.step()
                self.deform.optimizer.zero_grad(set_to_none=True)

        self.progress_bar.set_description(
            "Stage2 material training | EMA loss={}".format('%.7f' % self.ema_loss_for_log))
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

    parser.add_argument('--load_iter', type=int, default=-1, help="Iteration to load.")
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[4,35001, 1, 1000, 5000, 6000, 7_000] + list(range(8000, 300_0001, 2000)))
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[7_000, 10_000, 20_000, 30_000, 40000] + list(range(8000, 300_0001, 2000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform-type", type=str, default='mlp')

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    if not args.model_path.endswith(args.deform_type):
        args.model_path = os.path.join(os.path.dirname(os.path.normpath(args.model_path)),
                                       os.path.basename(os.path.normpath(args.model_path)) + f'_{args.deform_type}')

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    trainer = Trainer(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args),
              testing_iterations=args.test_iterations, saving_iterations=args.save_iterations,
              load_iter=args.load_iter)

    trainer.train(args.iterations)

    # All done
    print("\nTraining complete.")


