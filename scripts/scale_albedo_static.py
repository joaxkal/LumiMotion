
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
from gaussian_renderer.render_ir import render_ir
from torchvision.io import read_image
from utils.general_utils import safe_state, find_static_dataset
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel, DeformModel
import re
import torchvision
import json
from utils.graphics_utils import rgb_to_srgb, srgb_to_rgb

def render_set(dataset: ModelParams, pipeline: PipelineParams, load_iter, static_dataset):
    with torch.no_grad():
    
        dataset.eval = False # we do it to make sure we gather all possible cameras while data reading, probably could be removed. But final test cameras are ok.
        deform = DeformModel(deform_type=dataset.deform_type, is_blender=dataset.is_blender,
                             hyper_dim=dataset.hyper_dim,
                             pred_color=dataset.pred_color)
        deform_loaded = deform.load_weights(dataset.model_path, iteration=load_iter)

        gs_fea_dim = dataset.hyper_dim
        gaussians = GaussianModel(dataset.sh_degree, no_binary_separation=dataset.no_binary_separation,
                                  fea_dim=gs_fea_dim)

        scene = Scene(dataset, gaussians, load_iteration=load_iter)

        if not static_dataset:
            static_dataset = find_static_dataset(dataset.source_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        all_cameras = scene.getTestCameras() + scene.getTrainCameras() 
        assert len(all_cameras) > 0
        
        frame_static = re.search(r'timestep(\d+)', static_dataset).group(1) # extract timestep from folder name

        static_albedo_folder = os.path.join(static_dataset, "albedo")
        static_relight_folder = os.path.join(static_dataset, dataset.test_light_folder)

        view_static_pose = [c for c in all_cameras if frame_static.zfill(4) in c.image_path_train_light][0]
        fid_static = view_static_pose.fid.to("cuda")

        cameras = scene.getTestCameras()
        assert len(cameras) > 0

        render_path_parent = os.path.join(dataset.model_path, "scale_albedo_static_preview",
                                            "ours_{}".format(scene.loaded_iter))
        render_path = os.path.join(render_path_parent)
        makedirs(render_path, exist_ok=True)
          
        albedo_list = []
        albedo_gt_list = []


        for render_idx, view in enumerate(cameras[:]):

            if dataset.load2gpu_on_the_fly:
                view.load2device()

            N = gaussians.get_xyz.shape[0]

            time_input = fid_static.unsqueeze(0).expand(N, -1)
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
            rendering = torch.clamp(render_pkg_relight["base_color_linear"], 0.0, 1.0)
            torchvision.utils.save_image(rendering.clamp(0.0,1.0), os.path.join(render_path, 'albedo_train_static_linear{}'.format(render_idx) + ".png"))
            target_size = rendering.shape[1:]  # (H, W)

            view_path = view.image_path_train_light
            render_name = os.path.basename(view_path).split(".")[0]
            albedo_gt_path = [os.path.join(static_albedo_folder, f) for f in os.listdir(static_albedo_folder) if render_name in f][0]
            albedo_gt = read_image(albedo_gt_path).float()[:3]
            # Resize albedo_gt to match rendering resolution
            albedo_gt = F.interpolate(albedo_gt.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            albedo_gt /= 255.0  # normalize to [0,1]

            torchvision.utils.save_image(albedo_gt.clamp(0.0, 1.0), os.path.join(render_path, f'albedo_gt_static_srgb{render_idx}.png'))
            albedo_gt = srgb_to_rgb(albedo_gt)
            torchvision.utils.save_image(albedo_gt.clamp(0.0, 1.0), os.path.join(render_path, f'albedo_gt_static_linear{render_idx}.png'))

            #read mask from renders
            relight_gt_path = [os.path.join(static_relight_folder, f) for f in os.listdir(static_relight_folder) if render_name in f][0]
            mask = read_image(relight_gt_path).float()[3:4]
            # Resize mask to match rendering resolution
            mask = F.interpolate(mask.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            mask /= 255.0  # normalize to [0,1]
            # torchvision.utils.save_image(mask.clamp(0.0, 1.0), os.path.join(render_path, f'mask{render_idx}.png'))
        
            albedo_gt_list.append(albedo_gt.permute(1, 2, 0)[mask[0] > 0])
            albedo_list.append(rendering.permute(1, 2, 0)[mask[0] > 0])

            if dataset.load2gpu_on_the_fly:
                view.load2device("cpu")
                
        albedo_gts = torch.cat(albedo_gt_list, dim=0).cuda()
        albedo_ours = torch.cat(albedo_list, dim=0)
        albedo_scale_json = {}
        albedo_scale_json["0"] = [1.0, 1.0, 1.0]
        albedo_scale_json["1"] = [(albedo_gts/albedo_ours.clamp_min(1e-6))[..., 0].median().item()] * 3
        albedo_scale_json["2"] = (albedo_gts/albedo_ours.clamp_min(1e-6)).median(dim=0).values.tolist()
        albedo_scale_json["3"] = (albedo_gts/albedo_ours.clamp_min(1e-6)).mean(dim=0).tolist()
        print("Albedo scales:\n", albedo_scale_json)
            
        with open(os.path.join(args.model_path, "albedo_scale_linear_static.json"), "w") as f:
            json.dump(albedo_scale_json, f)


    

            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)

    parser.add_argument('--load_iter', type=int, default=-1, help="Iteration to load.")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--static_source_path", default="", type=str)




    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_set(model.extract(args), pipeline.extract(args), args.load_iter, args.static_source_path)