
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
import torch.nn.functional as F
from gaussian_renderer import render
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
    
        dataset.eval = True
        deform = DeformModel(deform_type=dataset.deform_type, is_blender=dataset.is_blender,
                             hyper_dim=dataset.hyper_dim,
                             pred_color=dataset.pred_color)
        deform_loaded = deform.load_weights(dataset.model_path, iteration=load_iter)

        gs_fea_dim = dataset.hyper_dim
        gaussians = GaussianModel(dataset.sh_degree, no_binary_separation=dataset.no_binary_separation,
                                 fea_dim=gs_fea_dim)

        scene = Scene(dataset, gaussians, load_iteration=load_iter)

        original_train_cameras = scene.getTrainCameras()
        all_timesteps = scene.all_timesteps

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # for synthetic scenes 5:6 train cameras have nice view
        # cameras = scene.getTrainCameras()[5:6]
        cameras = scene.getTestCameras()[:1]

        #for enerf actor 1 in paper we use cam 15.jpg :
        # cameras = [c for c in scene.getTrainCameras() if "15.jpg" in c.image_path_train_light][:1]

        assert len(cameras) > 0

        render_path_parent = os.path.join(dataset.model_path, "renders_stage1_insights",
                                            "ours_{}".format(scene.loaded_iter))
        render_path = os.path.join(render_path_parent)
        makedirs(render_path, exist_ok=True)

        for render_idx, view in enumerate(cameras):

            if dataset.load2gpu_on_the_fly:
                view.load2device()

            full_render_imgs = []
            alpha_imgs = []
            albedo_imgs = []
            small_gaussians_imgs = []
            separate_gaussians_imgs = []
            separate_gaussians_large_imgs = []
            normals_imgs = []

            render_name = view.image_name_train_light

            
            for interp_fid, timestep_idx in tqdm.tqdm(list(all_timesteps.items())[:]):

                N = gaussians.get_xyz.shape[0]

                time_input = interp_fid.unsqueeze(0).expand(N, -1)
                d_values = deform.step(gaussians.get_xyz.detach(), time_input, feature=gaussians.get_binary_feature(),
                                        camera_center=view.camera_center)
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], \
                                                                    d_values['d_scaling'], d_values['d_opacity'], \
                                                                    d_values['d_color']
                

                #full render
                render_pkg = render(view, gaussians, pipeline, background, \
                                    d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color, \
                                    )
                rendering = render_pkg["render"]

                torchvision.utils.save_image(rendering.clamp(0.0,1.0), os.path.join(render_path, 'full_t{}_cam{}'.format(timestep_idx, render_name) + ".png"))
                img_np = rendering.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                full_render_imgs.append(img_np)
                

                ## alpha render
                alpha_rend = render_pkg["rend_alpha"]
                # torchvision.utils.save_image(rendering.clamp(0.0,1.0), os.path.join(render_path, 'full_{}'.format(timestep_idx) + ".png"))
                img_np = alpha_rend.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                alpha_imgs.append(img_np)

                ## normals render
                norm_rend = render_pkg["rend_normal_view"]* 0.5 + 0.5
                torchvision.utils.save_image(norm_rend.clamp(0.0,1.0), os.path.join(render_path, 'normals_t{}_cam{}'.format(timestep_idx, render_name) + ".png"))
                img_np = norm_rend.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                normals_imgs.append(img_np)

                ## show small gaussians
                render_pkg = render(view, gaussians, pipeline, background, \
                                    d_xyz, d_rotation, 0, d_opacity=d_opacity, d_color=d_color, \
                                    clamp_scale_for_vis=True)
                rendering = render_pkg["render"]
                # torchvision.utils.save_image(rendering.clamp(0.0,1.0), os.path.join(render_path, 'gaussians_small_{}'.format(timestep_idx) + ".png"))
                img_np = rendering.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                small_gaussians_imgs.append(img_np)


                ##albedo- show color with no mlp-modifications and no sh1-3
                render_pkg = render(view, gaussians, pipeline, background, \
                                    d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=None, \
                                    override_color=gaussians.get_albedo)
                rendering = render_pkg["render"]
                # torchvision.utils.save_image(rendering.clamp(0.0,1.0), os.path.join(render_path, 'albedo_{}'.format(timestep_idx) + ".png"))
                img_np = rendering.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                albedo_imgs.append(img_np)


                ## show small gaussians - separation
                sep_color = torch.zeros_like(gaussians.get_xyz)
                sep_color[:, 0:1] = gaussians.get_binary_feature()
                sep_color[:, 1:2] = 1-gaussians.get_binary_feature()

                render_pkg = render(view, gaussians, pipeline, background, \
                                    d_xyz, d_rotation, 0, d_opacity=d_opacity, d_color=d_color, \
                                    clamp_scale_for_vis=True, override_color = sep_color)
                rendering = render_pkg["render"]
                torchvision.utils.save_image(rendering.clamp(0.0,1.0), os.path.join(render_path, 'separation_small_t{}_cam{}'.format(timestep_idx, render_name) + ".png"))
                img_np = rendering.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                separate_gaussians_imgs.append(img_np)

                ## show large gaussians - separation
                sep_color = torch.zeros_like(gaussians.get_xyz)
                sep_color[:, 0:1] = gaussians.get_binary_feature()
                sep_color[:, 1:2] = 1-gaussians.get_binary_feature()

                render_pkg = render(view, gaussians, pipeline, background, \
                                    d_xyz, d_rotation, 0, d_opacity=d_opacity, d_color=d_color, \
                                    clamp_scale_for_vis=False, override_color = sep_color)
                rendering = render_pkg["render"]
                torchvision.utils.save_image(rendering.clamp(0.0,1.0), os.path.join(render_path, 'separation_large_t{}_cam{}'.format(timestep_idx, render_name) + ".png"))
                img_np = rendering.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                separate_gaussians_large_imgs.append(img_np)
                
            if dataset.load2gpu_on_the_fly:
                    view.load2device('cpu')


            # Save videos
            vid_name = f"full_render_cam{render_name}"
            output_video_path = os.path.join(render_path, f"{vid_name}.mp4")
            writer = imageio.get_writer(output_video_path, fps=15)
            for img in full_render_imgs:
                cv2.putText(img, vid_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                writer.append_data(img)
            writer.close()

            vid_name = f"alpha_cam{render_name}"
            output_video_path = os.path.join(render_path, f"{vid_name}.mp4")
            writer = imageio.get_writer(output_video_path, fps=15)
            for img in alpha_imgs:
                cv2.putText(img, vid_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                writer.append_data(img)
            writer.close()

            vid_name = f"small_gaussians_cam{render_name}"
            output_video_path = os.path.join(render_path, f"{vid_name}.mp4")
            writer = imageio.get_writer(output_video_path, fps=15)
            for img in small_gaussians_imgs:
                cv2.putText(img, vid_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                writer.append_data(img)
            writer.close()

            vid_name = f"Albedo_cam{render_name}"
            output_video_path = os.path.join(render_path, f"{vid_name}.mp4")
            writer = imageio.get_writer(output_video_path, fps=15)
            for img in albedo_imgs:
                cv2.putText(img, vid_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                writer.append_data(img)
            writer.close()

            vid_name = f"Separation_cam{render_name}"
            output_video_path = os.path.join(render_path, f"{vid_name}.mp4")
            writer = imageio.get_writer(output_video_path, fps=15)
            for img in separate_gaussians_imgs:
                cv2.putText(img, vid_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                writer.append_data(img)
            writer.close()

            vid_name = f"Separation_large_cam{render_name}"
            output_video_path = os.path.join(render_path, f"{vid_name}.mp4")
            writer = imageio.get_writer(output_video_path, fps=15)
            for img in separate_gaussians_large_imgs:
                cv2.putText(img, vid_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                writer.append_data(img)
            writer.close()

            vid_name = f"Normals_cam{render_name}"
            output_video_path = os.path.join(render_path, f"{vid_name}.mp4")
            writer = imageio.get_writer(output_video_path, fps=15)
            for img in normals_imgs:
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