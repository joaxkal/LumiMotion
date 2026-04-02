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
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.point_utils import depth_to_normal
import numpy as np
import cv2
from utils.sh_utils import eval_sh, SH2RGB, RGB2SH

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, 
           d_xyz, d_rotation, d_scaling, d_opacity=None, d_color=None, scaling_modifier=1.0, 
           override_color=None, detach_xyz=False, 
           detach_scale=False, detach_rot=False, detach_opacity=False, scale_const=None, 
           d_rotation_bias=None, depth_filtering=False, 
           raster_settings_override = None, clamp_scale_for_vis = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    bg = bg_color

    if raster_settings_override:
        raster_settings = raster_settings_override
    else:

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz + d_xyz
    means2D = screenspace_points
    if scale_const is not None:
        opacity = torch.ones_like(pc.get_opacity)
    else:
        opacity = pc.get_opacity if d_opacity is None else pc.get_opacity + d_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier, d_rotation=None if type(d_rotation) is float else d_rotation, gs_rot_bias=d_rotation_bias)
    else:
        scales = pc.get_scaling + d_scaling
        rotations = pc.get_rotation_bias(d_rotation)
        if d_rotation_bias is not None:
            rotations = quaternion_multiply(d_rotation_bias, rotations)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if d_color is not None and type(d_color) is not float:
            shadowed_modulation = d_color[:,None, :3].clamp_max(1.0)
            final_color = RGB2SH(SH2RGB(pc.get_features[:, :1]) * (1 - shadowed_modulation))
            sh_features = torch.cat(
                    [
                        final_color,
                        pc.get_features[:, 1:],
                    ],
                    dim=1,)
        else:
            sh_features = pc.get_features
        
        if pipe.convert_SHs_python:
            shs_view = sh_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(sh_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = sh_features
    else:
        colors_precomp = override_color

    if detach_xyz:
        means3D = means3D.detach()
    if detach_rot or detach_scale:
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp.detach()
        else:
            rotations = rotations.detach() if detach_rot else rotations
            scales = scales.detach() if detach_scale else scales
    if detach_opacity:
        opacity = opacity.detach()

    if scale_const is not None:
        scales = scale_const * torch.ones_like(scales)

    if clamp_scale_for_vis:
        scales = scales.clamp_max(0.000005*pc.camera_extent)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    output = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )

    rendered_image, radii, allmap = output
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }


    #bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    whitebackground = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    if depth_filtering:
        if bg_color.equal(whitebackground):
            mask = (1-(torch.all(rendered_image >= 0.95, dim=0)).to(torch.int))
        else:
            mask = (1-(torch.all(rendered_image <= 0.05, dim=0)).to(torch.int))
    else:
        mask = 1
    render_alpha = allmap[1:2]

    # get normal map
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    render_normal = render_normal*mask

    # get normal map view space
    render_normal_view = allmap[2:5]
    render_normal_view = -render_normal_view*mask
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]
    render_dist = render_dist*mask

    # psedo surface attributes
    # surf depth is either median or expected by setting pipe.depth_ratio to 1 or 0
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    surf_depth = surf_depth*mask
    
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal, surf_point = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    surf_point = surf_point.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()
    surf_normal = surf_normal*mask

    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_normal_view': render_normal_view,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            'surf_point': surf_point,
            "bg_color": bg
    })

    return rets
