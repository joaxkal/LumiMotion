
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
import torch
import os
from os import makedirs
from gaussian_renderer.render_ir import render_ir
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel, DeformModel
import imageio
import cv2
import re
import tqdm
import numpy as np
import torchvision


def render_set(dataset: ModelParams, pipeline: PipelineParams, load_iter):
    with torch.no_grad():
    
        deform = DeformModel(deform_type=dataset.deform_type, is_blender=dataset.is_blender,
                             hyper_dim=dataset.hyper_dim,
                             pred_color=dataset.pred_color)
        deform_loaded = deform.load_weights(dataset.model_path, iteration=load_iter)

        gs_fea_dim = dataset.hyper_dim
        gaussians = GaussianModel(dataset.sh_degree, no_binary_separation=dataset.no_binary_separation,
                                  fea_dim=gs_fea_dim)

        scene = Scene(dataset, gaussians, load_iteration=load_iter)

        all_timesteps = scene.all_timesteps

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

 
        # for synthetic scenes 6:7 train cameras have nice view
        # cameras = scene.getTrainCameras()[6:7]
        cameras = scene.getTestCameras()[:1]

        #for enerf actor 1 in paper we use cam 15.jpg :
        # cameras = [c for c in scene.getTrainCameras() if "15.jpg" in c.image_path_train_light][:1]

        assert len(cameras) > 0

        render_path_parent = os.path.join(dataset.model_path, "trained_materials",
                                            "ours_{}".format(scene.loaded_iter))
        render_path = os.path.join(render_path_parent)
        makedirs(render_path, exist_ok=True)
          
        for render_idx, view in enumerate(cameras):

            if dataset.load2gpu_on_the_fly:
                view.load2device()

            albedo_render_imgs = []
            roughness_render_imgs = []
            
            for interp_fid, timestep_idx  in tqdm.tqdm(list(all_timesteps.items())[0:]):

                N = gaussians.get_xyz.shape[0]

                time_input = interp_fid.unsqueeze(0).expand(N, -1)
                d_values = deform.step(gaussians.get_xyz.detach(), time_input, feature=gaussians.get_binary_feature(),
                                        camera_center=view.camera_center)
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], \
                                                                    d_values['d_scaling'], d_values['d_opacity'], \
                                                                    d_values['d_color']
                

                render_pkg_relight = render_ir(viewpoint_camera=view, 
                                pc = scene.gaussians, pipe=pipeline, 
                                bg_color = background, d_xyz=d_xyz, 
                                d_rotation=d_rotation, d_scaling=d_scaling, 
                                d_opacity=d_opacity, d_color=d_color, 
                                relight=True, material_only=True)
                rendering = torch.clamp(render_pkg_relight["base_color"], 0.0, 1.0)
                rendered_roughness = torch.clamp(render_pkg_relight["roughness"], 0.0, 1.0)
                
                torchvision.utils.save_image(rendering.clamp(0.0, 1.0), os.path.join(render_path, 'albedo_cam{}_{}'.format(render_idx, timestep_idx) + ".png"))
                torchvision.utils.save_image(rendered_roughness.clamp(0.0, 1.0),
                                             os.path.join(render_path, 'roughness_cam{}_{}'.format(render_idx, timestep_idx) + ".png"))
                img_np = rendering.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                albedo_render_imgs.append(img_np)

                img_r_np = rendered_roughness.permute(1, 2, 0).cpu().numpy()
                img_r_np = (img_r_np * 255).clip(0, 255).astype(np.uint8)
                img_r_np = np.ascontiguousarray(img_r_np)
                roughness_render_imgs.append(img_r_np)
            
            if dataset.load2gpu_on_the_fly:
                view.load2device("cpu")

            # Save video
            vid_name = f"Albedo_cam{render_idx}"
            output_video_path = os.path.join(render_path, f"{vid_name}.mp4")
            writer = imageio.get_writer(output_video_path, fps=15)
            for img in albedo_render_imgs:
                cv2.putText(img, vid_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                writer.append_data(img)
            writer.close()

            # Save video
            vid_name = f"Roughness_cam{render_idx}"
            output_video_path = os.path.join(render_path, f"{vid_name}.mp4")
            writer = imageio.get_writer(output_video_path, fps=15)
            for img in roughness_render_imgs:
                cv2.putText(img, vid_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                writer.append_data(img)
            writer.close()

            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)

    parser.add_argument('--load_iter', type=int, default=-1, help="Iteration to load.")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")


    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_set(model.extract(args), pipeline.extract(args), args.load_iter)