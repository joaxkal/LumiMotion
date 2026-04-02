
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
from scene.light import EnvLight
from tqdm import tqdm

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

        # set EnvMap and sampling
        env_light = EnvLight(path=os.path.join(dataset.source_path, f"{args.test_light_folder}.hdr"), 
                             device='cuda',activation='none')
        env_light.build_mips()
        env_light.update_pdf()
        transform = torch.tensor([
            [0, -1, 0],
            [0, 0, 1],
            [-1, 0, 0]
        ], dtype=torch.float32, device="cuda")
        env_light.set_transform(transform)

        bg = 1 if dataset.white_background else 0
        background = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")

        dynamic_relight_folder = os.path.join(dataset.source_path, dataset.test_light_folder)

        cameras = scene.getTestCameras()
        assert len(cameras) > 0

        render_path_parent = os.path.join(dataset.model_path, "eval_relight_dynamic",
                                            "ours_{}".format(scene.loaded_iter))
        render_path = os.path.join(render_path_parent)
        makedirs(render_path, exist_ok=True)
          
        psnr_pbr = 0.0
        ssim_pbr = 0.0
        lpips_pbr = 0.0
        results_dict = {}

        build_bvh = True

        for render_idx, view in enumerate(tqdm(cameras[:], desc="Eval dynamic relight")):

            if dataset.load2gpu_on_the_fly:
                view.load2device()

            N = gaussians.get_xyz.shape[0]

            time_input = view.fid.unsqueeze(0).expand(N, -1)
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
                            relight=True, env_light=env_light, training=False, 
                            base_color_scale = base_color_scale,)
            

            target_size = render_pkg_relight["base_color_linear"].shape[1:]  # (H, W)

            view_path = view.image_path_train_light
            render_name = os.path.basename(view_path).split(".")[0]
            
            #read mask from renders
            relight_gt_path = [os.path.join(dynamic_relight_folder, f) for f in os.listdir(dynamic_relight_folder) if render_name in f][0]
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

                

            psnr_pbr += psnr(render_pkg_relight['render'], gt_image).mean().double().item()
            ssim_pbr += ssim(render_pkg_relight['render'], gt_image).mean().double().item()
            lpips_pbr += lpips(render_pkg_relight['render'], gt_image, net_type='vgg').mean().double().item()

            if dataset.load2gpu_on_the_fly:
                view.load2device("cpu")
                
    
                    
        psnr_pbr /= len(cameras)
        ssim_pbr /= len(cameras)
        lpips_pbr /= len(cameras)
        
        results_dict["psnr_pbr_avg"] = psnr_pbr
        results_dict["ssim_pbr_avg"] = ssim_pbr
        results_dict["lpips_pbr_avg"] = lpips_pbr

        print("\nEvaluating AVG: PSNR_PBR_RELIGHT {: .2f} SSIM_PBR_RELIGHT {: .3f} LPIPS_PBR_RELIGHT {: .3f}".format(psnr_pbr, ssim_pbr, lpips_pbr))
        with open(os.path.join(args.model_path, "results_relight_dynamic.json"), "w") as f:
            json.dump(results_dict, f, indent=4)
        print("Results saved to", os.path.join(args.model_path, "results_relight_dynamic.json"))
                    

            
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