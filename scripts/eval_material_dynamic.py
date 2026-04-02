
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
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel, DeformModel
import torchvision
import json
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr
from utils.graphics_utils import rgb_to_srgb, srgb_to_rgb


def render_set(dataset: ModelParams, pipeline: PipelineParams, load_iter):
    with torch.no_grad():

        with open(os.path.join(args.model_path, "albedo_scale_linear_dynamic.json"), "r") as f:
            albedo_scale_dict = json.load(f)
        base_color_scale = torch.tensor(albedo_scale_dict["2"], dtype=torch.float32, device="cuda")
    

        dataset.eval = False # we do it to make sure we gather all possible cameras while data reading, probably could be removed. But final test cameras are ok.
        deform = DeformModel(deform_type=dataset.deform_type, is_blender=dataset.is_blender,
                             hyper_dim=dataset.hyper_dim,
                             pred_color=dataset.pred_color)
        deform_loaded = deform.load_weights(dataset.model_path, iteration=load_iter)

        gs_fea_dim = dataset.hyper_dim
        gaussians = GaussianModel(dataset.sh_degree, no_binary_separation=dataset.no_binary_separation,
                                  fea_dim=gs_fea_dim)

        scene = Scene(dataset, gaussians, load_iteration=load_iter)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        dynamic_albedo_folder = os.path.join(dataset.source_path, "albedo")
        dynamic_relight_folder = os.path.join(dataset.source_path, dataset.test_light_folder)

        cameras = scene.getTestCameras()
        assert len(cameras) > 0

        render_path_parent = os.path.join(dataset.model_path, "eval_material_dynamic",
                                            "ours_{}".format(scene.loaded_iter))
        render_path = os.path.join(render_path_parent)
        makedirs(render_path, exist_ok=True)
          
        psnr_albedo = 0.0
        ssim_albedo = 0.0
        lpips_albedo = 0.0
        results_dict = {}


        for render_idx, view in enumerate(cameras[:]):

            if dataset.load2gpu_on_the_fly:
                view.load2device()

            N = gaussians.get_xyz.shape[0]

            time_input = view.fid.unsqueeze(0).expand(N, -1)
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
                            relight=True, material_only=True, 
                            base_color_scale = base_color_scale,)

            rendering = torch.clamp(render_pkg_relight["base_color_linear"], 0.0, 1.0)
            torchvision.utils.save_image(rendering.clamp(0.0,1.0), os.path.join(render_path, 'albedo_train_dynamic{}'.format(render_idx) + ".png"))
            target_size = rendering.shape[1:]  # (H, W)

            view_path = view.image_path_train_light
            render_name = os.path.basename(view_path).split(".")[0]
            albedo_gt_path = [os.path.join(dynamic_albedo_folder, f) for f in os.listdir(dynamic_albedo_folder) if render_name in f][0]
            albedo_gt = read_image(albedo_gt_path).float()[:3]
            # Resize albedo_gt to match rendering resolution
            albedo_gt = F.interpolate(albedo_gt.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0).cuda()
            albedo_gt /= 255.0  # normalize to [0,1]
            albedo_gt = srgb_to_rgb(albedo_gt)
            torchvision.utils.save_image(albedo_gt.clamp(0.0, 1.0), os.path.join(render_path, f'albedo_gt_dynamic{render_idx}.png'))

            #read mask from renders
            relight_gt_path = [os.path.join(dynamic_relight_folder, f) for f in os.listdir(dynamic_relight_folder) if render_name in f][0]
            mask = read_image(relight_gt_path).float()[3:4]
            # Resize mask to match rendering resolution
            mask = F.interpolate(mask.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0).cuda()
            mask /= 255.0  # normalize to [0,1]
            torchvision.utils.save_image(mask.clamp(0.0, 1.0), os.path.join(render_path, f'mask{render_idx}.png'))
        
            albedo_train = rendering * mask
            albedo_gt = albedo_gt * mask
            
            psnr_albedo += psnr(albedo_train, albedo_gt).mean().double().item()
            ssim_albedo += ssim(albedo_train, albedo_gt).mean().double().item()
            lpips_albedo += lpips(albedo_train, albedo_gt, net_type='vgg').mean().double().item()

            if dataset.load2gpu_on_the_fly:
                view.load2device("cpu")

                    
        psnr_albedo /= len(cameras)
        ssim_albedo /= len(cameras)
        lpips_albedo /= len(cameras)
        
        results_dict["psnr_albedo_avg"] = psnr_albedo
        results_dict["ssim_albedo_avg"] = ssim_albedo
        results_dict["lpips_albedo_avg"] = lpips_albedo

        print("\nEvaluating AVG: PSNR_ALBEDO {: .2f} SSIM_ALBEDO {: .3f} LPIPS_ALBEDO {: .3f}".format(psnr_albedo, ssim_albedo, lpips_albedo))
        with open(os.path.join(args.model_path, "results_material_dynamic.json"), "w") as f:
            json.dump(results_dict, f, indent=4)
        print("Results saved to", os.path.join(args.model_path, "results_material_dynamic.json"))
                    

            
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