#
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

from surfel_tracer import GaussianTracer
import torch
import numpy as np

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import build_scaling_rotation
from utils.sh_utils import *
from utils.general_utils import safe_normalize, flip_align_view
import trimesh


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


class GaussianModel:
    def __init__(self, sh_degree: int, no_binary_separation: bool, fea_dim=0, **kwargs):

        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans

        self.no_binary_separation = no_binary_separation
        
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.sh_env_degree = 2

        self._xyz = torch.empty(0)
        self._albedo_dc = torch.empty(0)
        self._albedo_rest = torch.empty(0)
        self._roughness = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)

        self.init_roughness_value = 0.75
        self.binarization_init_value = -1e-2


        self.fea_dim = fea_dim
        self.feature = torch.empty(0)

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.base_color_activation = lambda x: torch.sigmoid(x) * 0.94 + 0.03

        self.roughness_activation = torch.sigmoid
        self.inverse_roughness_activation = inverse_sigmoid

        # icosahedron, outer sphere radius is 1.0
        icosahedron = trimesh.creation.icosahedron()
        
        # change to inner sphere radius equal to 1.0
        # the central point of each face must be on the unit sphere
        self.unit_icosahedron_vertices = torch.from_numpy(icosahedron.vertices).float().cuda() * 1.2584 
        self.unit_icosahedron_faces = torch.from_numpy(icosahedron.faces).long().cuda()

        self.gaussian_tracer = GaussianTracer(transmittance_min=0.03)
        self.FG_LUT = torch.from_numpy(
            np.fromfile("assets/bsdf_256_256.bin", dtype=np.float32).reshape(1, 256, 256, 2)).cuda()

        self.alpha_min = 1 / 100 #255 #100 slightly faster,but check if quality still ok 
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


    @property
    def get_envmap(self):
        return self.env_map
    
        
    def param_names(self):
        return ['_xyz',
                '_albedo_dc',
                '_albedo_rest',
                '_scaling', 
                '_rotation', 
                '_opacity', 
                'max_radii2D', 
                'xyz_gradient_accum']

    @classmethod
    def build_from(cls, gs, **kwargs):
        new_gs = GaussianModel(**kwargs)
        new_gs._xyz = nn.Parameter(gs._xyz)

        new_gs._albedo_dc = nn.Parameter(gs._albedo_dc)
        new_gs._albedo_rest = nn.Parameter(gs._albedo_rest)

        new_gs._scaling = nn.Parameter(gs._scaling)
        new_gs._rotation = nn.Parameter(gs._rotation)
        new_gs._opacity = nn.Parameter(gs._opacity)
        new_gs.feature = nn.Parameter(gs.feature)
        new_gs.max_radii2D = torch.zeros((new_gs.get_xyz.shape[0]), device="cuda")
        return new_gs

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    def get_rotation_bias(self, rotation_bias=None):
        rotation_bias = rotation_bias if rotation_bias is not None else 0.
        return self.rotation_activation(self._rotation + rotation_bias)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_rough(self):
        return self.roughness_activation(self._roughness)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_albedo(self):
        return torch.clamp(SH2RGB(self._albedo_dc.squeeze()), 0.03, 0.97)


    @property
    def get_features(self):
        "returns color for phase 1. In phase 2. we will use only albedo_dc to make it diffuse. We will retrain albedo anyway "
        albedo_dc = self._albedo_dc
        albedo_rest = self._albedo_rest
        return torch.cat((albedo_dc, albedo_rest), dim=1)

    def get_binary_feature(self, eval=True, T=None):
        #T is not really a param, we fix it to 0.5
        # you can unlock this to test different values, we hardcoded it to avoid bugs

        T = 0.5

        if self.no_binary_separation:
            # tmp_log_feat = self.binarization_init_value*torch.ones_like(self.feature)
            # return nn.functional.sigmoid((1/T)*(tmp_log_feat))
            return torch.ones_like(self.feature)
        
        log_feat = self.feature
        
        if eval:
            u = torch.ones(log_feat.shape, device=log_feat.device)*0.5
        else:
            u = torch.rand(log_feat.shape, device=log_feat.device)
        binary_feature = nn.functional.sigmoid((1/T)*(log_feat+torch.log(u) - torch.log(1-u)))
        return binary_feature
    
    def compute_env_sh(self):
        return self.env_params

    def get_covariance(self, scaling_modifier=1, xyz=None, scales=None, rotation=None, gs_rot_bias=None):
        return self.covariance_activation(xyz, scales, scaling_modifier, rotation)
    

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float=5., print_info=True, max_point_num=150_000):
        self.camera_extent = spatial_lr_scale #why they hardcoded it?
        self.spatial_lr_scale = 5
        if type(pcd.points) == np.ndarray:
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        else:
            fused_point_cloud = pcd.points
        if type(pcd.colors) == np.ndarray:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        else:
            fused_color = pcd.colors


        albedo = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()


        if print_info:
            print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._albedo_dc = nn.Parameter(albedo[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._albedo_rest = nn.Parameter(albedo[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        roughness = self.inverse_roughness_activation(torch.full_like(opacities, self.init_roughness_value))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._roughness = nn.Parameter(roughness.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # init for unsupervised bianrization
        self.feature = nn.Parameter(self.binarization_init_value * torch.ones([self._xyz.shape[0], self.fea_dim], dtype=torch.float32).to("cuda:0"), requires_grad=True)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.spatial_lr_scale = 5

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._albedo_dc], 'lr': training_args.albedo_lr, "name": "albedo_dc"},
            {'params': [self._albedo_rest], 'lr': training_args.albedo_rest_lr, "name": "albedo_rest"},

            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        #try -except here, because in first training we dont need albedo_dc_stage1 variable
        try:
            l.append(
                {'params': [self._albedo_dc_stage1], 'lr': training_args.albedo_after_stage1_lr, "name": "albedo_dc_stage1"}
            )
        except:
            pass

        if self.fea_dim >0:
            l.append(
                {'params': [self.feature], 'lr': training_args.feature_lr, 'name': 'feature'}
            )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale, lr_final=training_args.position_lr_final * self.spatial_lr_scale, lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._albedo_dc.shape[1]*self._albedo_dc.shape[2]):
            l.append('albedo_dc_{}'.format(i))
        for i in range(self._albedo_dc.shape[1]*self._albedo_dc.shape[2]):
            l.append('albedo_dc_stage1_{}'.format(i))
        for i in range(self._albedo_rest.shape[1]*self._albedo_rest.shape[2]):
            l.append('albedo_rest_{}'.format(i))
        for i in range(self._albedo_dc.shape[1]*self._albedo_dc.shape[2]):
            l.append('f_dc_{}'.format(i))

        l.append('opacity')
        l.append('roughness')

        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self.fea_dim):
            l.append('fea_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        roughness = self._roughness.detach().cpu().numpy()
        albedo_dc = self._albedo_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        #Try-except, because in first training we dont need _albedo_dc_stage1 variable
        try:
            albedo_dc_stage1 = self._albedo_dc_stage1.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        except:
            albedo_dc_stage1 = self._albedo_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        albedo_rest = self._albedo_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, albedo_dc, albedo_dc_stage1, albedo_rest, albedo_dc, opacities, roughness, scale, rotation), axis=1)
        if self.fea_dim > 0:
            feature = self.feature.detach().cpu().numpy()
            attributes = np.concatenate((attributes, feature), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def reset_static_dynamic(self):
        # Not thouroughly used, but maybe worth exploring
        feature_new = torch.ones_like(self.feature) * 0.001
        optimizable_tensors = self.replace_tensor_to_optimizer(feature_new, "feature")
        self.feature = optimizable_tensors["feature"]

    def load_ply(self, path, og_number_points=-1):
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        #@TODO remove try-except
        try:
            roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
        except:
            roughness = self.inverse_roughness_activation(torch.tensor(np.full_like(opacities, self.init_roughness_value)))


        albedo_dc = np.zeros((xyz.shape[0], 3, 1))
        albedo_dc[:, 0, 0] = np.asarray(plydata.elements[0]["albedo_dc_0"])
        albedo_dc[:, 1, 0] = np.asarray(plydata.elements[0]["albedo_dc_1"])
        albedo_dc[:, 2, 0] = np.asarray(plydata.elements[0]["albedo_dc_2"])

        #TODO remove try-except. We keep it here because stage1 didnt need albedo_dc_stage1 var for training
        try:
            albedo_dc_stage1 = np.zeros((xyz.shape[0], 3, 1))
            albedo_dc_stage1[:, 0, 0] = np.asarray(plydata.elements[0]["albedo_dc_stage1_0"])
            albedo_dc_stage1[:, 1, 0] = np.asarray(plydata.elements[0]["albedo_dc_stage1_1"])
            albedo_dc_stage1[:, 2, 0] = np.asarray(plydata.elements[0]["albedo_dc_stage1_2"])
        except:
            albedo_dc_stage1 = albedo_dc.copy()

        extra_albedo_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("albedo_rest")]
        extra_albedo_names = sorted(extra_albedo_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_albedo_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        albedo_extra = np.zeros((xyz.shape[0], len(extra_albedo_names)))
        for idx, attr_name in enumerate(extra_albedo_names):
            albedo_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        albedo_extra = albedo_extra.reshape((albedo_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_0") or p.name.startswith("scale_1")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        fea_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("fea")]
        feas = np.zeros((xyz.shape[0], self.fea_dim))
        for idx, attr_name in enumerate(fea_names):
            feas[:, idx] = np.asarray(plydata.elements[0][attr_name])



        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._albedo_dc = nn.Parameter(torch.tensor(albedo_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._albedo_dc_stage1 = nn.Parameter(torch.tensor(albedo_dc_stage1, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        self._albedo_rest = nn.Parameter(torch.tensor(albedo_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        if self.fea_dim > 0:
            self.feature = nn.Parameter(torch.tensor(feas, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._albedo_dc = optimizable_tensors["albedo_dc"]
        self._albedo_rest = optimizable_tensors["albedo_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._roughness = optimizable_tensors["roughness"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.fea_dim > 0:
            self.feature = optimizable_tensors["feature"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def densification_postfix(self, new_xyz, new_albedo_dc, new_albedo_rest, new_opacities, new_roughness, new_scaling, new_rotation, new_feature=None):
        d = {"xyz": new_xyz,
            "albedo_dc": new_albedo_dc,
            "albedo_rest": new_albedo_rest,
            "roughness": new_roughness,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation}
        
        if self.fea_dim > 0:
            d["feature"] = new_feature

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._albedo_dc = optimizable_tensors["albedo_dc"]
        self._albedo_rest = optimizable_tensors["albedo_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._roughness = optimizable_tensors["roughness"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.fea_dim > 0:
            self.feature = optimizable_tensors["feature"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads=None, grad_threshold=None, scene_extent=None, N=2, selected_pts_mask=None, without_prune=False):
        if selected_pts_mask is None:
            n_init_points = self.get_xyz.shape[0]
            # Extract points that satisfy the gradient condition
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grads.shape[0]] = grads.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling,
                                                            dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        #stds2 = torch.cat([stds[:,0].unsqueeze(1), stds], dim=1) 
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)

        new_albedo_dc = self._albedo_dc[selected_pts_mask].repeat(N,1, 1)
        new_albedo_rest = self._albedo_rest[selected_pts_mask].repeat(N,1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)

        new_feature = self.feature[selected_pts_mask].repeat(N, 1) if self.fea_dim > 0 else None
        
        self.densification_postfix(new_xyz, new_albedo_dc, new_albedo_rest, new_opacity, new_roughness, new_scaling, new_rotation, new_feature)

        if not without_prune:
            prune_filter = torch.cat(
                (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
            self.prune_points(prune_filter)

    def densify_and_clone(self, grads=None, grad_threshold=None, scene_extent=None, selected_pts_mask=None):
        # Extract points that satisfy the gradient condition
        if selected_pts_mask is None:
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling,
                                                            dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_albedo_dc = self._albedo_dc[selected_pts_mask]
        new_albedo_rest = self._albedo_rest[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_feature = self.feature[selected_pts_mask] if self.fea_dim > 0  else None

        self.densification_postfix(new_xyz,
                                   new_albedo_dc, new_albedo_rest, new_opacities, new_roughness, new_scaling, new_rotation, new_feature)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    def get_boundings(self, d_rotation, d_xyz, d_scaling, alpha_min=0.01):
        mu = self.get_xyz+d_xyz
        opacity = self.get_opacity
        scale = self.get_scaling+d_scaling
        scale = torch.cat([scale, torch.full_like(scale, 1e-6)], dim=-1)
        
        L = build_scaling_rotation(scale, self._rotation+d_rotation)
        
        vertices_b = (2 * (opacity/alpha_min).log()).sqrt()[:, None] * (self.unit_icosahedron_vertices[None] @ L.transpose(-1, -2)) + mu[:, None]
        faces_b = self.unit_icosahedron_faces[None] + torch.arange(mu.shape[0], device="cuda")[:, None, None] * 12
        gs_id = torch.arange(mu.shape[0], device="cuda")[:, None].expand(-1, faces_b.shape[1])
        return vertices_b.reshape(-1, 3), faces_b.reshape(-1, 3), gs_id.reshape(-1)
    

    def build_bvh(self, d_rotation, d_xyz, d_scaling):
        vertices_b, faces_b, gs_id = self.get_boundings(alpha_min=self.alpha_min,
                                                        d_rotation=d_rotation, d_xyz=d_xyz, d_scaling=d_scaling)
        self.gaussian_tracer.build_bvh(vertices_b, faces_b, gs_id)
        
    def update_bvh(self,  d_rotation, d_xyz, d_scaling):
        vertices_b, faces_b, gs_id = self.get_boundings(alpha_min=self.alpha_min,
                                                        d_rotation=d_rotation, d_xyz=d_xyz, d_scaling=d_scaling)
        self.gaussian_tracer.update_bvh(vertices_b, faces_b, gs_id)
        
    def trace(self, rays_o, rays_d, features=None, camera_center=None, 
              xyz=None, scales = None, rotation=None, opacity=None,
              back_culling=False, shs=None, use_zeros=False):
        
        means3D = xyz
        if shs is None:
            shs = self.get_features
        if use_zeros:
            shs = torch.zeros_like(self.get_features).cuda()
        opacity = opacity
        
        #@TODO check if this nan to num is Ok
        epsilon = 1e-7
        s = 1.0 / (scales + epsilon * (scales == 0).float())
        R = build_rotation(rotation) #build_rotation(self._rotation)
        ru = R[:, :, 0] * s[:,0:1]
        rv = R[:, :, 1] * s[:,1:2]
        
        splat2world = self.get_covariance(xyz=xyz, scales=scales, rotation=rotation)
        normals_raw = splat2world[: ,2, :3] 
        if camera_center is not None:
            normals_raw, positive = flip_align_view(normals_raw, means3D - camera_center)
        normals = safe_normalize(normals_raw)
        
        color, normal, feature, depth, alpha = self.gaussian_tracer.trace(rays_o, rays_d, means3D, opacity, ru, rv, normals, features, shs, alpha_min=self.alpha_min, deg=self.active_sh_degree, back_culling=back_culling)
        
        alpha_ = alpha[..., None]
        color = torch.where(alpha_ < 1 - self.gaussian_tracer.transmittance_min, color, color / alpha_)
        normal = torch.where(alpha_ < 1 - self.gaussian_tracer.transmittance_min, normal, normal / alpha_)
        feature = torch.where(alpha_ < 1 - self.gaussian_tracer.transmittance_min, feature, feature / alpha_)
        depth = torch.where(alpha < 1 - self.gaussian_tracer.transmittance_min, depth, depth / alpha)
        alpha = torch.where(alpha < 1 - self.gaussian_tracer.transmittance_min, alpha, torch.ones_like(alpha))
        
        return {
            "color": color,
            "normal": normal,
            "feature": feature,
            "depth": depth,
            "alpha" : alpha,
            "normals": normals,
        }

