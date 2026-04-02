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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import torch.nn.functional as F

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, 
                 image_train_light, gt_alpha_mask, image_name_train_light, 
                 image_path_train_light,
                 uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", fid=None, depth=None,
                 proj_matrix = None):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name_train_light = image_name_train_light

        self.image_path_train_light = image_path_train_light


        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image_train_light = image_train_light.clamp(0.0, 1.0).to(self.data_device)

        self.fid = torch.Tensor(np.array([fid])).to(self.data_device)
        self.image_width = self.original_image_train_light.shape[2]
        self.image_height = self.original_image_train_light.shape[1]
        self.depth = torch.Tensor(depth).to(self.data_device) if depth is not None else None
        self.gt_alpha_mask = gt_alpha_mask

        if gt_alpha_mask is not None:
            self.gt_alpha_mask = self.gt_alpha_mask.to(self.data_device)
            # self.original_image *= gt_alpha_mask.to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).to(self.data_device)
        
        if torch.is_tensor(proj_matrix):
             # to do move it to graphic utils!
            self.projection_matrix = proj_matrix.to(self.data_device)
        
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        self.camera_center = self.world_view_transform.inverse()[3, :3]

        v, u = torch.meshgrid(torch.arange(self.image_height, device=self.data_device),
                              torch.arange(self.image_width, device=self.data_device), indexing="ij")
        focal_x = self.image_width / (2 * np.tan(self.FoVx * 0.5))
        focal_y = self.image_height / (2 * np.tan(self.FoVy * 0.5))
        rays_d_camera = torch.stack([(u - self.image_width / 2 + 0.5) / focal_x,
                                  (v - self.image_height / 2 + 0.5) / focal_y,
                                  torch.ones_like(u)], dim=-1).reshape(-1, 3)
        rays_d = rays_d_camera @ self.world_view_transform[:3, :3].T
        self.rays_d_unnormalized = rays_d
        self.rays_d = F.normalize(rays_d, dim=-1)
        self.rays_o = self.camera_center[None].expand_as(self.rays_d)
        self.rays_rgb = self.original_image_train_light.permute(1, 2, 0).reshape(-1, 3)
        self.rays_d_hw = self.rays_d.reshape(self.image_height, self.image_width, 3)
        self.rays_d_hw_unnormalized = rays_d.reshape(self.image_height, self.image_width, 3)

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def load2device(self, data_device='cuda'):
        self.original_image_train_light = self.original_image_train_light.to(data_device)
        self.world_view_transform = self.world_view_transform.to(data_device)
        self.projection_matrix = self.projection_matrix.to(data_device)
        self.full_proj_transform = self.full_proj_transform.to(data_device)
        self.camera_center = self.camera_center.to(data_device)
        self.fid = self.fid.to(data_device)
        self.rays_d_unnormalized = self.rays_d_unnormalized.to(data_device)
        self.rays_d = self.rays_d.to(data_device)
        self.rays_o = self.rays_o.to(data_device)
        self.rays_rgb = self.rays_rgb.to(data_device)
        self.rays_d_hw = self.rays_d_hw.to(data_device)
        self.rays_d_hw_unnormalized = self.rays_d_hw_unnormalized.to(data_device)
    
    def get_rays(self):
        return self.rays_o, self.rays_d
        
    def get_rays_rgb(self):
        return self.original_image.permute(1, 2, 0).reshape(-1, 3)
        


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
