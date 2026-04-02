
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


import torch.nn.functional as F
import torch
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer.render_ir import render_ir
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import numpy as np
from scene import Scene, GaussianModel, DeformModel
import torchvision
from scene.light import EnvLight
import imageio
import cv2
from scipy.ndimage import gaussian_filter
import json

def render_set(dataset: ModelParams, pipeline: PipelineParams, load_iter, colmap_convention, hdr_filepath):
    
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

        N = gaussians.get_xyz.shape[0]

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # for synthetic scenes 6:7 train cameras have nice view
        # cameras = scene.getTrainCameras()[6:7]
        cameras = scene.getTestCameras()[:1]

        #for enerf actor 1 in paper we use cam 15.jpg :
        # cameras = [c for c in scene.getTrainCameras() if "15.jpg" in c.image_path_train_light][:1]


        assert len(cameras) > 0

        env_light = EnvLight(path=hdr_filepath, device='cuda', activation="none")
        env_light.build_mips()
        env_light.update_pdf()
        transform = torch.tensor([
            [0, -1, 0],
            [0, 0, 1],
            [-1, 0, 0]
        ], dtype=torch.float32, device="cuda")

        if colmap_convention:  #enerf has colmap
            colmap_rot = torch.tensor([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]
            ], dtype=torch.float32, device="cuda")
            transform = transform @ colmap_rot
        env_light.set_transform(transform)

        hdr_name = os.path.splitext(os.path.basename(hdr_filepath))[0]

        render_path_parent = os.path.join(dataset.model_path, "render_relight_with_hdr_{}".format(hdr_name), "ours_{}".format(scene.loaded_iter))
        render_path = os.path.join(render_path_parent)
        makedirs(render_path, exist_ok=True)

        albedo_scale_path = os.path.join(dataset.model_path, "albedo_scale_linear_dynamic.json")
        if os.path.isfile(albedo_scale_path):
            with open(albedo_scale_path, "r") as f:
                albedo_scale_dict = json.load(f)
            base_color_scale = torch.tensor(albedo_scale_dict.get("2"), dtype=torch.float32, device="cuda")
        else:
            base_color_scale = None

        built_bvh = False
        for render_idx, view in enumerate(cameras[:]):

            if dataset.load2gpu_on_the_fly:
                view.load2device()

            timesteps = list(all_timesteps.items())

            # Get the first and last timestep (as (key, value) pairs)
            first = timesteps[0] 
            last = timesteps[-1]

            selected_timesteps = timesteps #timesteps or [first, last] or [first]...

            render_imgs =[]
            for interp_fid, timestep_idx in tqdm(selected_timesteps, desc="generate image for one timestep"):

                time_input = interp_fid.unsqueeze(0).expand(N, -1)
                d_values = deform.step(gaussians.get_xyz.detach(), time_input, feature=gaussians.get_binary_feature(), camera_center=view.camera_center)
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], \
                                                                    d_values['d_scaling'], d_values['d_opacity'], \
                                                                    d_values['d_color']

                if not built_bvh:
                    gaussians.build_bvh(d_rotation=d_rotation, d_xyz=d_xyz, d_scaling = d_scaling) 
                    built_bvh = True
                else:
                    gaussians.update_bvh(d_rotation=d_rotation, d_xyz=d_xyz, d_scaling = d_scaling) 
                

                result_pkg = render_ir(view, gaussians, pipeline, background, \
                                    d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color, \
                                    relight=True, env_light=env_light, \
                                    training=False, base_color_scale=base_color_scale, skip_tracer=False)
                                
                rendering = result_pkg["render"]
                torchvision.utils.save_image(rendering.clamp(0.0,1.0), os.path.join(render_path, 'illum_{}_t{}'.format(render_idx, timestep_idx) + ".png"))
                img_np = rendering.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                render_imgs.append(img_np)
            
            if dataset.load2gpu_on_the_fly:
                view.load2device("cpu")


            # Save videos
            vid_name = "render_relight_with_hdr{}_view{}".format(pipeline.diffuse_sample_num, render_idx)
            output_video_path = os.path.join(render_path, f"{vid_name}.mp4")
            writer = imageio.get_writer(output_video_path, fps=15)
            for img in render_imgs:
                #cv2.putText(img, vid_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                writer.append_data(img)
            writer.close()
            print(f"Video saved as {output_video_path}")

            
if __name__ == "__main__":

    """
    This script requires substantial cleaning. 
    For now, please manually adjust lit pixels, timesteps and cameras you need to render.
    """
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)

    parser.add_argument('--load_iter', type=int, default=-1, help="Iteration to load.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--colmap_convention", action="store_true", default=False)
    parser.add_argument("--hdr", type=str, help="Path to hdr.")
    


    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_set(model.extract(args), pipeline.extract(args), args.load_iter, args.colmap_convention, args.hdr)