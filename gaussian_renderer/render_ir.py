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
from utils.sh_utils import SH2RGB, RGB2SH
from utils.point_utils import depth_to_normal
from utils.graphics_utils import rgb_to_srgb, srgb_to_rgb
import numpy as np
from utils.relight_utils import sample_incident_rays
import cv2
import torch.nn.functional as F
import nvdiffrast.torch as dr


def render_ir(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
              d_xyz, d_rotation, d_scaling, d_opacity=None, d_color=None, scaling_modifier=1.0,
              override_color=None, random_bg_color=False, detach_xyz=False,
              detach_scale=False, detach_rot=False, detach_opacity=False, scale_const=None,
              d_rotation_bias=None, depth_filtering=False, raster_settings_override=None, opt=None,
              training=False, relight=False, env_light=None, base_color_scale=None, material_only=False, rot_env_x=None,
              colmap_transform=False, override_dirs=None, skip_tracer=False):
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

    bg = bg_color if not random_bg_color else torch.rand_like(bg_color)

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
        cov3D_precomp = pc.get_covariance(scaling_modifier,
                                          d_rotation=None if type(d_rotation) is float else d_rotation,
                                          gs_rot_bias=d_rotation_bias)
    else:
        scales = pc.get_scaling + d_scaling
        rotations = pc.get_rotation_bias(d_rotation)
    

    # here compute color. Our albedo is in sRGB -we have pseudo-prior from stage1
    base_color = srgb_to_rgb(pc.get_albedo)
    override_color = base_color

    shs = None
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


    # TODO one rasterizer for multiple features! We know there are such implementations, but we only tested ours.
    #render albedo
    output = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    #render roughness
    output_rough = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=pc.get_rough.repeat(1, 3),
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    #render color from stage 1
    shadowed_modulation = d_color[:,None, :3].clamp_max(1.0)
    sh_dc_mlp = RGB2SH(SH2RGB(pc._albedo_dc_stage1[:, :1]) * (1 - shadowed_modulation))
    sh_features = torch.cat((sh_dc_mlp, pc._albedo_rest), dim=1)
    output_sh = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=sh_features,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )

    # irgs will have different outputs #here
    rendered_image, radii, allmap = output
    rendered_base_color = rendered_image
    rendered_roughness, _, _ = output_rough
    rendered_image_sh, _, _ = output_sh

    # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    whitebackground = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    if depth_filtering:
        if bg_color.equal(whitebackground):
            mask = (1 - (torch.all(rendered_image >= 0.95, dim=0)).to(torch.int))
        else:
            mask = (1 - (torch.all(rendered_image <= 0.05, dim=0)).to(torch.int))
    else:
        mask = 1
    render_alpha = torch.nan_to_num(allmap[1:2], 0, 0)

    # get normal map
    render_normal = torch.nan_to_num(allmap[2:5], 0, 0)
    render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_view_transform[:3, :3].T)).permute(2, 0,
                                                                                                                 1)
    render_normal = render_normal * mask

    # get normal map view space
    render_normal_view = torch.nan_to_num(allmap[2:5], 0, 0)
    render_normal_view = -render_normal_view * mask

    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = torch.nan_to_num(allmap[0:1], 0, 0)
    render_depth_expected = (render_depth_expected / render_alpha.clamp_min(1e-6))
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # # get expected depth map - old version, breaks grad
    # render_depth_expected = allmap[0:1]
    # render_depth_expected = render_depth_expected / render_alpha
    # render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # get depth distortion map
    render_dist = allmap[6:7]
    render_dist = render_dist * mask

    # psedo surface attributes
    # surf depth is either median or expected by setting pipe.depth_ratio to 1 or 0

    # surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    ratio_test = min(0.5, pipe.depth_ratio)
    surf_depth = render_depth_expected * (1 - ratio_test) + (ratio_test) * render_depth_median

    surf_depth = surf_depth * mask

    # surf points should be also returned by depth_to_normal(viewpoint_camera, surf_depth), but I do exactly like in IRGS here
    points = surf_depth.permute(1, 2, 0) * viewpoint_camera.rays_d_hw_unnormalized + viewpoint_camera.camera_center

    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal, surf_point = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2, 0, 1)
    surf_point = surf_point.permute(2, 0, 1)
    # points = surf_point.permute(1, 2, 0) #wrong!

    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()
    surf_normal = surf_normal * mask

    # Use normal map computed in 2DGS pipeline to perform reflection query
    normal_map = render_normal.permute(1, 2, 0)
    normal_map = normal_map / render_alpha.permute(1, 2, 0).clamp_min(1e-6)
    normal_map = F.normalize(normal_map, dim=-1)

    # rendered_base_color, rendered_roughness = rendered_features.split([3, 1], dim=0)
    if base_color_scale is not None:
        rendered_base_color = rendered_base_color * base_color_scale[:, None, None]

    if material_only:
        results = {
            "roughness": rendered_roughness*render_alpha,
            "base_color": rgb_to_srgb(rendered_base_color)*render_alpha + bg_color[:, None, None] * (1 - render_alpha),
            "base_color_linear": rendered_base_color*render_alpha + bg_color[:, None, None] * (1 - render_alpha),
            "viewspace_points": means2D,
            "visibility_filter": radii > 0,
            "radii": radii,
            ## normal, accum alpha, dist, depth map
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_normal_view': render_normal_view,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            'surf_point': surf_point,
            "bg_color": bg,
        }
        return results

    if training:
        if opt.train_ray:
            mask_alpha = render_alpha[0] > 0.9  # 0.0 originally
            mask_sum = mask_alpha.sum()

            num_pixels = opt.trace_num_rays // (pipe.diffuse_sample_num + pipe.light_sample_num)
            if num_pixels > mask_sum:
                ray_ids = torch.arange(mask_sum, device='cuda')
            else:
                ray_ids = torch.multinomial(torch.ones(mask_sum, device=mask_sum.device), num_pixels, replacement=False)

            mask_ = mask_alpha[mask_alpha]
            mask_[ray_ids] = False
            mask = torch.zeros_like(mask_alpha)
            mask[mask_alpha] = ~mask_
        else:
            mask = render_alpha[0] > 0
    else:
        mask = render_alpha[0] > 0

    rays_d = viewpoint_camera.rays_d_hw
    w_o = -rays_d

    if training:
        render_results = rendering_equation(rendered_base_color.permute(1, 2, 0)[mask],
                                            rendered_roughness.permute(1, 2, 0)[mask],
                                            normal_map[mask], points[mask], w_o[mask], pc,
                                            pipe=pipe, training=training, relight=relight,
                                            camera_center=viewpoint_camera.camera_center,
                                            envlight=env_light, xyz=means3D, scales=scales, sh_features=sh_features, rotation=rotations,
                                            opacity=opacity, colmap_transform=colmap_transform,
                                            override_dirs=None, rot_env_x=None, skip_tracer=False)
    else:
        render_results = rendering_equation_chunk(rendered_base_color.permute(1, 2, 0)[mask],
                                                  rendered_roughness.permute(1, 2, 0)[mask],
                                                  normal_map[mask], points[mask], w_o[mask], pc,
                                                  pipe=pipe, training=training, relight=relight,
                                                  camera_center=viewpoint_camera.camera_center,
                                                  envlight=env_light, xyz=means3D, scales=scales, sh_features=sh_features, rotation=rotations,
                                                  opacity=opacity, rot_env_x=rot_env_x, colmap_transform=colmap_transform,
                                                  override_dirs=override_dirs, skip_tracer=skip_tracer)

    diffuse = render_results['diffuse']
    specular = render_results['specular']
    light_direct = render_results['light_direct']

    rendered_diffuse = torch.zeros_like(rendered_image).permute(1, 2, 0)
    rendered_diffuse[mask] = diffuse
    rendered_diffuse = rendered_diffuse.permute(2, 0, 1)

    rendered_specular = torch.zeros_like(rendered_image).permute(1, 2, 0)
    rendered_specular[mask] = specular
    rendered_specular = rendered_specular.permute(2, 0, 1)
    rendered_full = rgb_to_srgb(rendered_diffuse + rendered_specular)
    final_image = rendered_full * render_alpha + bg_color[:, None, None] * (1 - render_alpha)

    final_image_sh = rendered_image_sh + bg_color[:, None, None] * (1 - render_alpha)

    # direct_lights = rgb_to_srgb(pc.get_envmap(rays_d, mode='pure_env').permute(2,0,1))
    direct_lights = rgb_to_srgb(env_light(rays_d, mode='pure_env').permute(2, 0, 1))
    env_only = direct_lights

    results = {
        "render": final_image,
        "env_only": env_only,
        "render_sh": final_image_sh,
        "diffuse": rgb_to_srgb(rendered_diffuse),
        "specular": rgb_to_srgb(rendered_specular),
        "mask": mask,
        "roughness": rendered_roughness * render_alpha,
        "base_color": rgb_to_srgb(rendered_base_color) * render_alpha + bg_color[:, None, None] * (1 - render_alpha),
        "base_color_linear": rendered_base_color * render_alpha + bg_color[:, None, None] * (1 - render_alpha),
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
        ## normal, accum alpha, dist, depth map
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_normal_view': render_normal_view,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        "ray_light_direct": light_direct
    }

    if opt is not None and training and opt.train_ray:
        alpha = render_alpha.permute(1, 2, 0)[mask]
        full = diffuse + specular
        full = rgb_to_srgb(full)
        ray_rgb = full * alpha + bg_color[None, :] * (1 - alpha)

        light_indirect = render_results['light_indirect']
        rendered_light_indirect = torch.zeros_like(rendered_image).permute(1, 2, 0)
        rendered_light_indirect[mask] = light_indirect
        rendered_light_indirect = rendered_light_indirect.permute(2, 0, 1) * render_alpha

        rendered_light_direct = torch.zeros_like(rendered_image).permute(1, 2, 0)
        rendered_light_direct[mask] = light_direct
        rendered_light_direct = rendered_light_direct.permute(2, 0, 1) * render_alpha

        results.update({
            "ray_rgb": ray_rgb,
            "light_direct": rgb_to_srgb(rendered_light_direct),
            "light_indirect": rgb_to_srgb(rendered_light_indirect),
        })

    if not training:
        visibility = render_results['visibility']
        light = render_results['light']
        light_indirect = render_results['light_indirect']

        rendered_visibility = torch.zeros_like(rendered_image[:1]).permute(1, 2, 0)
        rendered_visibility[mask] = visibility
        rendered_visibility = rendered_visibility.permute(2, 0, 1) * render_alpha

        rendered_light = torch.zeros_like(rendered_image).permute(1, 2, 0)
        rendered_light[mask] = light
        rendered_light = rendered_light.permute(2, 0, 1) * render_alpha

        rendered_light_indirect = torch.zeros_like(rendered_image).permute(1, 2, 0)
        rendered_light_indirect[mask] = light_indirect
        rendered_light_indirect = rendered_light_indirect.permute(2, 0, 1) * render_alpha

        rendered_light_direct = torch.zeros_like(rendered_image).permute(1, 2, 0)
        rendered_light_direct[mask] = light_direct
        rendered_light_direct = rendered_light_direct.permute(2, 0, 1) * render_alpha

        final_image_env = rendered_full * render_alpha + direct_lights * (1 - render_alpha)

        results.update({
            "render_env": final_image_env,
            "light_direct": rgb_to_srgb(rendered_light_direct),
            "visibility": rendered_visibility,
            "light": rgb_to_srgb(rendered_light),
            "light_indirect": rgb_to_srgb(rendered_light_indirect),
        })

    # visibility = render_results['visibility']
    # light_blocked = render_results['light_blocked']
    # results.update({
    #         "visibility": visibility,
    #         "light_blocked": light_blocked
    #     })

    return results


def rendering_equation_chunk(base_color, roughness, normal,
                             position, w_o, pc, pipe, envlight=None,
                             training=False, f0=0.02, relight=False,
                             chunk_size=2 ** 20, camera_center=None,
                             image_sh=None, xyz=None, scales=None, sh_features=None, rotation=None,
                             opacity=None, **kwargs):
    chunk_size = chunk_size // (pipe.diffuse_sample_num + pipe.light_sample_num)
    if base_color.shape[0] <= chunk_size:
        return rendering_equation(base_color, roughness, normal, position,
                                  w_o, pc, pipe, training, f0, relight=relight,
                                  camera_center=camera_center, sh_features=sh_features,
                                  envlight=envlight, xyz=xyz, scales=scales,
                                  rotation=rotation, opacity=opacity, **kwargs)
    else:
        results = []
        for i in range(0, base_color.shape[0], chunk_size):
            results.append(rendering_equation(base_color[i:i + chunk_size],
                                              roughness[i:i + chunk_size],
                                              normal[i:i + chunk_size],
                                              position[i:i + chunk_size],
                                              w_o[i:i + chunk_size],
                                              pc, pipe, training, f0, relight=relight,
                                              camera_center=camera_center,
                                              sh_features=sh_features,
                                              envlight=envlight,
                                              xyz=xyz, scales=scales, rotation=rotation,
                                              opacity=opacity, **kwargs))

        return {k: torch.cat([r[k] for r in results], 0) for k in results[0]}


def rendering_equation(base_color, roughness, normals, position, viewdirs, pc, pipe, training=False, f0=0.04,
                       relight=False, camera_center=None, envlight=None, xyz=None, scales=None, sh_features=None,
                       rotation=None, opacity=None, **kwargs):
    # print(pipe.diffuse_sample_num)
    B = base_color.shape[0]

    if pipe.diffuse_sample_num > 0 and pipe.light_sample_num == 0:
        incident_dirs, incident_areas = sample_incident_rays(normals, training, pipe.diffuse_sample_num)
    elif pipe.diffuse_sample_num > 0 and pipe.light_sample_num > 0:
        raise NotImplemented
        # It actually is implemented correctly, but for our small envmaps doesnt make sense so I commented it out for safety
        p_diffuse = pipe.diffuse_sample_num / (pipe.diffuse_sample_num + pipe.light_sample_num)
        p_light = pipe.light_sample_num / (pipe.diffuse_sample_num + pipe.light_sample_num)

        diffuse_directions, diffuse_areas = sample_incident_rays(normals, training, pipe.diffuse_sample_num)
        diffuse_pdfs = 1 / diffuse_areas

        light_directions, light_pdfs = envlight.sample_light_directions(B, pipe.light_sample_num, training)

        diffuse_pdfs_light = 1 / (2 * np.pi)
        light_pdfs_diffuse = envlight.light_pdf(diffuse_directions)

        diffuse_pdfs = diffuse_pdfs * p_diffuse + light_pdfs_diffuse * p_light
        light_pdfs = diffuse_pdfs_light * p_diffuse + light_pdfs * p_light

        incident_dirs = torch.cat([diffuse_directions, light_directions], dim=1)
        incident_pdfs = torch.cat([diffuse_pdfs, light_pdfs], dim=1)
        incident_areas = 1 / incident_pdfs.clamp_min(1e-6)
    else:
        raise NotImplementedError

    if kwargs["override_dirs"] is not None:
        incident_dirs = kwargs["override_dirs"].repeat(normals.shape[0], 1, 1)  # Shape (N, 1, 3)

    global_incident_lights = envlight(incident_dirs, mode='pure_env')

    if relight:
        #here get albedo is in srgb (like in stage1)
        features = torch.cat([pc.get_albedo, pc.get_rough.cuda()], dim=1)

        if kwargs["skip_tracer"]:
            #SHOULD BE USED ONLY FOR DEBUG! 
            # its quicker on A100 so we can see if there are some artifacts when relit without tracer
            trace_outputs = {"alpha":torch.zeros(normals.shape[0], pipe.diffuse_sample_num, device="cuda"),
                            "feature":torch.ones(normals.shape[0], pipe.diffuse_sample_num, 4, device="cuda"),
                            "normal":torch.ones(normals.shape[0], pipe.diffuse_sample_num, 3, device="cuda")}
        else:
            trace_outputs = pc.trace(position.unsqueeze(1) + incident_dirs * pipe.light_t_min, incident_dirs,
                                    features=features, camera_center=camera_center,
                                    xyz=xyz, scales=scales, rotation=rotation, opacity=opacity)  #added for dynamic

        trace_alpha = trace_outputs['alpha'][..., None]
        incident_visibility = 1 - trace_alpha
        trace_feature = trace_outputs['feature'] / trace_alpha.clamp_min(1e-6)
        trace_normal = F.normalize(trace_outputs['normal'], dim=-1)
        trace_base_color_srgb, trace_roughness = trace_feature.split([3, 1], dim=-1)

        #here albedo was in srgb (like in stage1), convert to linear rgb
        trace_base_color = srgb_to_rgb(trace_base_color_srgb)

        if pipe.wo_indirect_relight:
            print("Without indirect in relight")
            trace_diffuse = torch.zeros_like(trace_base_color)
            trace_specular = torch.zeros_like(trace_base_color)
        else:
            trace_diffuse = trace_base_color * envlight(trace_normal, mode='diffuse')
            trace_wi = -incident_dirs
            trace_NdotV = (trace_normal * trace_wi).sum(-1, keepdim=True)
            trace_reflected = F.normalize(trace_NdotV * trace_normal * 2 - trace_wi, dim=-1)
            fg_uv = torch.cat([trace_NdotV, trace_roughness], -1).clamp(0, 1)
            fg = dr.texture(pc.FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear", boundary_mode="clamp").reshape(*fg_uv.shape)
            trace_specular = envlight(trace_reflected, roughness=trace_roughness, mode='specular') * (f0 * fg[..., 0:1] + fg[..., 1:2])

        local_incident_lights = (trace_diffuse + trace_specular) * trace_alpha
        
    else:
        # here IRGS uses rgb color (sh_features arg) from first stage to get indirect light. Its a smart way.
        trace_outputs = pc.trace(position.unsqueeze(1) + incident_dirs * pipe.light_t_min, incident_dirs,
                                 xyz=xyz, scales=scales, rotation=rotation, features=None,
                                 camera_center=camera_center, opacity=opacity, shs = sh_features)
        trace_alpha = trace_outputs['alpha'][..., None]
        incident_visibility = 1 - trace_outputs['alpha'][..., None]
        local_incident_lights = srgb_to_rgb(trace_outputs['color']) #because colors from Stage1 were optimized as srgb
        if pipe.wo_indirect:
            local_incident_lights = torch.zeros_like(local_incident_lights)
        if pipe.detach_indirect:
            incident_visibility = incident_visibility.detach()
            local_incident_lights = local_incident_lights.detach()
    incident_lights = incident_visibility * global_incident_lights + local_incident_lights

    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)
    f_d = base_color[:, None] / np.pi  # *0+1, for quick vis only illum
    f_s = GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=0.04)

    transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    diffuse = ((f_d) * transport).mean(dim=-2)
    specular = ((f_s) * transport).mean(dim=-2)

    if pipe.wo_specular:
        specular *=0

    if training:
        results = {
            "diffuse": diffuse,
            "specular": specular,
            "light_direct": global_incident_lights.mean(dim=1),
            "light_indirect": local_incident_lights.mean(dim=1),
            "visibility": incident_visibility.mean(dim=1),
            "light_blocked": (trace_alpha * global_incident_lights).mean(dim=1),
        }
    else:
        results = {
            "diffuse": diffuse,
            "specular": specular,
            "visibility": incident_visibility.mean(dim=1),
            "light": (transport).mean(dim=-2) / 3.14,  # incident_lights.mean(dim=1),
            "light_direct": global_incident_lights.mean(dim=1),
            "light_indirect": local_incident_lights.mean(dim=1),
            "light_blocked": (trace_alpha * global_incident_lights).mean(dim=1),
        }
    return results


def GGX_specular(
        normal,
        pts2c,
        pts2l,
        roughness,
        fresnel
):
    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
    N = N * NoV.sign()  # [nrays, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1] TODO check broadcast
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

    alpha = roughness * roughness  # [nrays, 3]
    alpha2 = alpha * alpha  # [nrays, 3]
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 6.98316) * VoH
    frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]

    frac = frac0 * alpha2[:, None, :]  # [nrays, 1]
    nom0 = NoH * NoH * (alpha2[:, None, :] - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k[:, None, :]) + k[:, None, :]
    nom = (4 * np.pi * nom0 * nom0 * nom1[:, None, :] * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom
    return spec


