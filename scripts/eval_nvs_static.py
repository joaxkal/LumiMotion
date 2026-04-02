
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
from scene.light import EnvLight
import os
from os import makedirs
import torch.nn.functional as F
from gaussian_renderer.render_ir import render_ir
from torchvision.io import read_image
from utils.general_utils import safe_state, find_static_dataset
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene, GaussianModel, DeformModel
import re
import torchvision
import json
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr
from tqdm import tqdm

def render_set(dataset: ModelParams, pipeline: PipelineParams, opt: OptimizationParams, load_iter, static_dataset):
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
        
       
        env_light = EnvLight(path=None, device='cuda',
                             resolution=[opt.envmap_resolution // 2, opt.envmap_resolution],
                             max_res=opt.envmap_resolution, activation=opt.envmap_activation)
        env_light.load_weights(dataset.model_path, scene.loaded_iter)
        

        bg = 1 if dataset.white_background else 0
        background = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")

        all_cameras = scene.getTestCameras() + scene.getTrainCameras() 
        assert len(all_cameras) > 0

        frame_static = re.search(r'timestep(\d+)', static_dataset).group(1) # extract timestep from folder name

        static_relight_folder = os.path.join(static_dataset, dataset.train_light_folder)

        view_static_pose = [c for c in all_cameras if frame_static.zfill(4) in c.image_path_train_light][0]
        fid_static = view_static_pose.fid.to("cuda")

        cameras = scene.getTestCameras()
        assert len(cameras) > 0

        render_path_parent = os.path.join(dataset.model_path, "eval_nvs_static",
                                            "ours_{}".format(scene.loaded_iter))
        render_path = os.path.join(render_path_parent)
        makedirs(render_path, exist_ok=True)
          
        psnr_nvs = 0.0
        ssim_nvs = 0.0
        lpips_nvs = 0.0
        results_dict = {}

        build_bvh = True

        for render_idx, view in enumerate(tqdm(cameras[:], desc="Eval static nvs")):

            if dataset.load2gpu_on_the_fly:
                view.load2device()

            N = gaussians.get_xyz.shape[0]

            time_input = fid_static.unsqueeze(0).expand(N, -1)
            d_values = deform.step(gaussians.get_xyz.detach(), time_input, feature=gaussians.get_binary_feature(),
                                    camera_center=view.camera_center)
            d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], \
                                                                d_values['d_scaling'], d_values['d_opacity'], \
                                                                d_values['d_color']

            
            if build_bvh:
                gaussians.build_bvh(d_rotation = d_rotation, d_xyz=d_xyz, d_scaling = d_scaling)
                build_bvh = False

            render_pkg_relight = render_ir(viewpoint_camera=view, 
                            pc = scene.gaussians, pipe=pipeline, 
                            bg_color = background, d_xyz=d_xyz, 
                            d_rotation=d_rotation, d_scaling=d_scaling, 
                            d_opacity=d_opacity, d_color=d_color, 
                            relight=False, env_light=env_light, training=False)
            

            target_size = render_pkg_relight["base_color_linear"].shape[1:]  # (H, W)

            view_path = view.image_path_train_light
            render_name = os.path.basename(view_path).split(".")[0]
            
            #read mask from renders
            relight_gt_path = [os.path.join(static_relight_folder, f) for f in os.listdir(static_relight_folder) if render_name in f][0]
            mask = read_image(relight_gt_path).float()[3:4]
            # Resize mask to match rendering resolution
            mask = F.interpolate(mask.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0).cuda()
            mask /= 255.0  # normalize to [0,1]
            torchvision.utils.save_image(mask.clamp(0.0, 1.0), os.path.join(render_path, f'mask{render_idx}.png'))

            # read gt images (duplicated code!)
            gt_image = read_image(relight_gt_path).float()[:3]
            gt_image = F.interpolate(gt_image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0).cuda()
            gt_image /= 255.0  # normalize to [0,1]
            torchvision.utils.save_image(gt_image.clamp(0.0, 1.0), os.path.join(render_path, f'gt_image{render_idx}.png'))


            render_pkg_relight["render"] = render_pkg_relight["render"] * mask + (1 - mask) * bg
            torchvision.utils.save_image(render_pkg_relight["render"].clamp(0.0, 1.0), os.path.join(render_path, f'render{render_idx}.png'))

            gt_image_env = gt_image + render_pkg_relight["env_only"] * (1 - mask)
            torchvision.utils.save_image(gt_image_env.clamp(0.0, 1.0), os.path.join(render_path, f'gt_image_env{render_idx}.png'))

                

            psnr_nvs += psnr(render_pkg_relight['render'], gt_image).mean().double().item()
            ssim_nvs += ssim(render_pkg_relight['render'], gt_image).mean().double().item()
            lpips_nvs += lpips(render_pkg_relight['render'], gt_image, net_type='vgg').mean().double().item()

            if dataset.load2gpu_on_the_fly:
                view.load2device("cpu")
                
    
                    
        psnr_nvs /= len(cameras)
        ssim_nvs /= len(cameras)
        lpips_nvs /= len(cameras)
        
        results_dict["psnr_nvs_avg"] = psnr_nvs
        results_dict["ssim_nvs_avg"] = ssim_nvs
        results_dict["lpips_nvs_avg"] = lpips_nvs

        print("\nEvaluating AVG: PSNR_NVS {: .2f} SSIM_NVS {: .3f} LPIPS_NVS {: .3f}".format(psnr_nvs, ssim_nvs, lpips_nvs))
        with open(os.path.join(args.model_path, "results_nvs_static.json"), "w") as f:
            json.dump(results_dict, f, indent=4)
        print("Results saved to", os.path.join(args.model_path, "results_nvs_static.json"))
                    

            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)


    parser.add_argument('--load_iter', type=int, default=-1, help="Iteration to load.")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--static_source_path", default="", type=str)


    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_set(model.extract(args), pipeline.extract(args), opt.extract(args), args.load_iter, args.static_source_path)