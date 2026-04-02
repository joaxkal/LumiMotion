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
from scene import Scene
import uuid
from utils.image_utils import psnr, lpips, alex_lpips
from utils.image_utils import ssim as ssim_func
from piq import LPIPS
lpips = LPIPS()
from argparse import Namespace
from pytorch_msssim import ms_ssim
from utils.normal_utils import compute_normal_world_space
from utils.general_utils import colormap
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
from gaussian_renderer.render_ir import render_ir

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, 
                       renderFunc, renderArgs, deform, load2gpu_on_the_fly, progress_bar):
        
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr_reconstruct = 0.0
    test_ssim_reconstruct = 0.0
    test_lpips_reconstruct = 1e10
    test_ms_ssim_reconstruct = 0.0
    test_alex_lpips_reconstruct = 1e10


    if iteration in testing_iterations:
        
        torch.cuda.empty_cache()

        # we make full evaluation only for last testing iteration. Otherwise on subset of test cameras

        selected_test_cams = scene.getTestCameras() if iteration==testing_iterations[-1] else []
        validation_configs = ({'name': 'test', 'cameras': selected_test_cams}, #scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(0,30,10)]})


        for config in validation_configs:

            if config['cameras'] and len(config['cameras']) > 0:
                
                psnr_list_reconstruct, ssim_list_reconstruct, lpips_list_reconstruct, l1_list_reconstruct = [], [], [], []
                ms_ssim_list_reconstruct, alex_lpips_list_reconstruct = [], []

                for idx, viewpoint in enumerate(config['cameras']):
                    
                    #if not full eval needed
                    if idx > 3:
                        break
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz

                    if deform.name == 'mlp':
                        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    else:
                        time_input = 0

                    d_values = deform.step(xyz.detach(), time_input, 
                                           feature=scene.gaussians.get_binary_feature(), 
                                           is_training=False, 
                                           camera_center=viewpoint.camera_center)
                    
                    d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], \
                        d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
                    
                    gt_image_train_light = torch.clamp(viewpoint.original_image_train_light.to("cuda"), 0.0, 1.0)

                    
                    render_pkg_reconstruct = renderFunc(viewpoint, scene.gaussians, *renderArgs, 
                                                        d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, 
                                                        d_opacity=d_opacity, d_color=d_color)
                
                    image_reconstruct = torch.clamp(render_pkg_reconstruct["render"], 0.0, 1.0)

                    # append reconstruction for metrics
                    l1_list_reconstruct.append(l1_loss(image_reconstruct[None], gt_image_train_light[None]).mean())
                    psnr_list_reconstruct.append(psnr(image_reconstruct[None], gt_image_train_light[None]).mean())
                    ssim_list_reconstruct.append(ssim_func(image_reconstruct[None], gt_image_train_light[None], data_range=1.).mean())
                    lpips_list_reconstruct.append(lpips(image_reconstruct[None], gt_image_train_light[None]).mean())
                    ms_ssim_list_reconstruct.append(ms_ssim(image_reconstruct[None], gt_image_train_light[None], data_range=1.).mean())
                    alex_lpips_list_reconstruct.append(alex_lpips(image_reconstruct[None], gt_image_train_light[None]).mean())


                    #get normals in world space
                    quaternions = scene.gaussians.get_rotation_bias(d_rotation)
                    scales = scene.gaussians.get_scaling + d_scaling 
                    xyz = scene.gaussians.get_xyz + d_xyz

                    normal_vectors, multiplier = compute_normal_world_space(
                        quaternions, scales, viewpoint.world_view_transform, scene.gaussians.get_xyz)
                    
                    #max 5 images for tensorboard plots
                    if tb_writer and (idx < 5):

                        rgb_precomp_albedo = scene.gaussians.get_albedo
                        render_pkg_albedo = renderFunc(viewpoint, scene.gaussians, *renderArgs, 
                                                        d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling,
                                                        d_opacity=d_opacity, d_color=None,
                                                        override_color=rgb_precomp_albedo)
                        image_albedo = torch.clamp(render_pkg_albedo["render"], 0.0, 1.0)



                        tb_writer.add_images(config['name'] + "_view_{}_{}/albedo".format(viewpoint.image_name_train_light, idx), image_albedo[None], global_step=iteration)
                        
                        # if iteration == testing_iterations[0]:
                        tb_writer.add_images(config['name'] + "_view_{}_{}/ground_truth_train_light".format(viewpoint.image_name_train_light, idx), gt_image_train_light[None], global_step=iteration)
                        
                        # render reconstruct
                        tb_writer.add_images(config['name'] + "_view_{}_{}/reconstruct".format(viewpoint.image_name_train_light, idx), image_reconstruct[None], global_step=iteration)


                        # render depth 
                        depth = render_pkg_reconstruct["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}_{}/depth".format(viewpoint.image_name_train_light, idx), depth[None], global_step=iteration)

                        #rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                        #check ours normals - should be exactly the same
                        normals_precomp = (normal_vectors*0.5 + 0.5)
                        render_pkg_albedo = renderFunc(viewpoint, scene.gaussians,*renderArgs, 
                                                d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling,
                                                d_opacity=d_opacity, d_color=d_color,
                                                override_color=normals_precomp)
                        rend_normal = render_pkg_albedo["render"]
                        tb_writer.add_images(config['name'] + "_view_{}_{}/rend_normal".format(viewpoint.image_name_train_light, idx), rend_normal[None], global_step=iteration)
                        
                        # render surfel normals
                        surf_normal = render_pkg_albedo["surf_normal"] * 0.5 + 0.5
                        tb_writer.add_images(config['name'] + "_view_{}_{}/surf_normal".format(viewpoint.image_name_train_light, idx), surf_normal[None], global_step=iteration)
                        
                        #render distance
                        rend_dist = render_pkg_albedo["rend_dist"]
                        rend_dist = colormap(rend_dist.cpu().numpy()[0])
                        tb_writer.add_images(config['name'] + "_view_{}_{}/rend_dist".format(viewpoint.image_name_train_light, idx), rend_dist[None], global_step=iteration)
                        
                        rend_alpha = render_pkg_albedo['rend_alpha']
                        tb_writer.add_images(config['name'] + "_view_{}_{}/rend_alpha".format(viewpoint.image_name_train_light, idx), rend_alpha[None], global_step=iteration)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')

                #### PRINT AGGREGATED METRICS
                #reconstruct
                l1_test_reconstruct = torch.stack(l1_list_reconstruct).mean()
                psnr_test_reconstruct = torch.stack(psnr_list_reconstruct).mean()
                ssim_test_reconstruct = torch.stack(ssim_list_reconstruct).mean()
                lpips_test_reconstruct = torch.stack(lpips_list_reconstruct).mean()
                ms_ssim_test_reconstruct = torch.stack(ms_ssim_list_reconstruct).mean()
                alex_lpips_test_reconstruct = torch.stack(alex_lpips_list_reconstruct).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr_reconstruct = psnr_test_reconstruct
                    test_ssim_reconstruct = ssim_test_reconstruct
                    test_lpips_reconstruct = lpips_test_reconstruct
                    test_ms_ssim_reconstruct = ms_ssim_test_reconstruct
                    test_alex_lpips_reconstruct = alex_lpips_test_reconstruct
                
                print("\n[ITER {}] Evaluating reconstruct {}: L1 {:.2f} PSNR {:.2f} SSIM {:.2f} LPIPS {:.2f} MS SSIM {:.2f} ALEX_LPIPS {:.2f}".format(
                    iteration, config['name'], 
                    l1_test_reconstruct, psnr_test_reconstruct, 
                    ssim_test_reconstruct, lpips_test_reconstruct, 
                    ms_ssim_test_reconstruct, alex_lpips_test_reconstruct
                ))
                

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/reconstruct - l1_loss', l1_test_reconstruct, iteration)
                    tb_writer.add_scalar(config['name'] + '/reconstruct - psnr', test_psnr_reconstruct, iteration)
                    tb_writer.add_scalar(config['name'] + '/reconstruct - ssim', test_ssim_reconstruct, iteration)
                    tb_writer.add_scalar(config['name'] + '/reconstruct - lpips', test_lpips_reconstruct, iteration)
                    tb_writer.add_scalar(config['name'] + '/reconstruct - ms-ssim', test_ms_ssim_reconstruct, iteration)
                    tb_writer.add_scalar(config['name'] + '/reconstruct - alex-lpips', test_alex_lpips_reconstruct, iteration)
                

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr_reconstruct, test_ssim_reconstruct, test_lpips_reconstruct, test_ms_ssim_reconstruct, test_alex_lpips_reconstruct


def training_report_relight_screen_space(tb_writer, iteration, Ll1, loss, elapsed, testing_iterations,
                                         scene: Scene,
                                         renderArgs, deform, load2gpu_on_the_fly, env_light):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        if tb_writer:
            env_dict = env_light.render_env_map()
            env2_map = env_dict["env2"].permute(2, 0, 1)
            env2_map_unscaled = env2_map.clone()
            env2_map_scaled = env2_map / torch.clamp(torch.max(env2_map), min=1e-8)

            tb_writer.add_images(
                "envmap/train_clamp",
                torch.clamp(env2_map_unscaled, 0, 1).unsqueeze(0),
                global_step=iteration
            )
            tb_writer.add_images(
                "envmap/train_scaled",
                env2_map_scaled.unsqueeze(0),
                global_step=iteration
            )

        torch.cuda.empty_cache()
        selected_test_cams = scene.getTestCameras() if iteration == testing_iterations[-1] else []
        selected_train_cams = [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(0, 30, 10)]
        validation_configs = ({'name': 'test', 'cameras': selected_test_cams},
                              {'name': 'train', 'cameras': selected_train_cams})

        pipeline, background = renderArgs
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                for idx, viewpoint in enumerate(config['cameras']):
                    if idx > 3:
                        break
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()

                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz

                    if deform.name == 'mlp':
                        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    else:
                        time_input = 0

                    d_values = deform.step(xyz.detach(), time_input,
                                           feature=scene.gaussians.get_binary_feature(),
                                           is_training=False,
                                           camera_center=viewpoint.camera_center)

                    d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], \
                                                                       d_values['d_rotation'], d_values['d_scaling'], \
                                                                       d_values['d_opacity'], d_values['d_color']

                    scene.gaussians.update_bvh(d_rotation=d_rotation, d_xyz=d_xyz, d_scaling=d_scaling)

                    render_pkg_relight = render_ir(viewpoint_camera=viewpoint,
                                                   pc=scene.gaussians, pipe=pipeline,
                                                   bg_color=background, d_xyz=d_xyz,
                                                   d_rotation=d_rotation, d_scaling=d_scaling,
                                                   d_opacity=d_opacity, d_color=d_color,
                                                   relight=True,
                                                   material_only=True)

                    if tb_writer and (idx < 5):
                        image_albedo = torch.clamp(render_pkg_relight["base_color"], 0.0, 1.0)
                        image_roughness = torch.clamp(render_pkg_relight["roughness"], 0.0, 1.0)

                        tb_writer.add_images(
                            config['name'] + "_view_{}_{}/albedo".format(viewpoint.image_name_train_light, idx),
                            image_albedo[None], global_step=iteration)
                        tb_writer.add_images(
                            config['name'] + "_view_{}_{}/roughness".format(viewpoint.image_name_train_light, idx),
                            image_roughness[None], global_step=iteration)
                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

