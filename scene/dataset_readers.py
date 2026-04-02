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
import sys
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from glob import glob
import cv2 as cv
from pathlib import Path
from smplx.body_models import SMPLX
from plyfile import PlyData, PlyElement
from .SMCReader import SMCReader
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import open3d as o3d
from tqdm import tqdm
from utils.graphics_utils import getProjectionMatrix_DNA


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array

    image_train_light: np.array
    image_path_train_light: str
    image_name_train_light: str

    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None
    proj_matrix: Optional[torch.Tensor] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    all_timesteps: Optional[list] = None


def getNerfppNorm(cam_info, apply=False):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    cam_centers = []
    if apply:
        c2ws = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        if apply:
            c2ws.append(C2W)
        cam_centers.append(C2W[:3, 3:4])
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal
    translate = -center
    if apply:
        c2ws = np.stack(c2ws, axis=0)
        c2ws[:, :3, -1] += translate
        c2ws[:, :3, -1] /= radius
        w2cs = np.linalg.inv(c2ws)
        for i in range(len(cam_info)):
            cam = cam_info[i]
            cam_info[i] = cam._replace(R=w2cs[i, :3, :3].T, T=w2cs[i, :3, 3])
        apply_translate = translate
        apply_radius = radius
        translate = 0
        radius = 1.
        return {"translate": translate, "radius": radius, "apply_translate": apply_translate, "apply_radius": apply_radius}
    else:
        return {"translate": translate, "radius": radius}


def translate_cam_info(cam_info, translate):
    for i in range(len(cam_info)):
        cam = cam_info[i]
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        C2W[:3, 3] += translate
        W2C = np.linalg.inv(C2W)
        cam_info[i] = cam._replace(R=W2C[:3, :3].T, T=W2C[:3, 3])


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                        vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly_from_gaussians(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['f_dc_0'], vertices['f_dc_1'],
                        vertices['f_dc_2']]).T / 255.0
    
    try:
        normals = np.vstack([vertices['normal_x'], vertices['normal_y'], vertices['normal_z']]).T
        pcd = BasicPointCloud(points=positions, colors=colors, normals=normals)
    except:
        shs = np.random.random((len(positions), 3)) / 255.0
        pcd = BasicPointCloud(points=positions, colors=shs, normals=np.zeros((len(positions), 3)))

    return pcd




def readCamerasFromTransforms(path, transformsfile, train_light, white_background, extension=".png", no_bg=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        frames = sorted(frames, key=lambda x: int(os.path.basename(x['file_path']).split('.')[0].split('_')[-1]))
        for idx, frame in enumerate(frames):
            if frame["file_path"].endswith('jpg') or frame["file_path"].endswith('png'):
                cam_name_train_light = os.path.join(path, train_light, frame["file_path"])
            else:
                cam_name_train_light = os.path.join(path, train_light, frame["file_path"] + extension)
            if 'time' in frame:
                frame_time = frame['time']
            else:
                frame_time = idx / len(frames)
            if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.join(path, train_light, frame["file_path"]))), 'rgba')):
                cam_name_train_light = os.path.join(os.path.dirname(os.path.dirname(os.path.join(path, train_light, frame["file_path"]))), 'rgba', os.path.basename(frame['file_path'])).replace('.jpg', '.png')

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            image_path_train_light = os.path.join(path, cam_name_train_light)
            image_name_train_light = Path(cam_name_train_light).stem
            image_train_light = Image.open(image_path_train_light)

            ### im train light preprocessing
            im_data_train_light = np.array(image_train_light.convert("RGBA"))

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data_train_light = im_data_train_light / 255.0
            mask = norm_data_train_light[..., 3:4]

            arr_train_light = norm_data_train_light[:, :, :3] 
            if no_bg:
                norm_data_train_light[:, :, :3] = norm_data_train_light[:, :, 3:4] * norm_data_train_light[:, :, :3] + bg * (1 - norm_data_train_light[:, :, 3:4])
            
            arr_train_light = np.concatenate([arr_train_light, mask], axis=-1)

            image_train_light = Image.fromarray(np.array(arr_train_light * 255.0, dtype=np.byte), "RGBA" if arr_train_light.shape[-1] == 4 else "RGB")

            fovy = focal2fov(fov2focal(fovx, image_train_light.size[0]), image_train_light.size[1])
            FovY = fovx
            FovX = fovy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, 
                                        image_train_light=image_train_light, image_path_train_light=image_path_train_light, image_name_train_light=image_name_train_light, 
                                        width=image_train_light.size[0], height=image_train_light.size[1], fid=frame_time))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png", no_bg=True, args=None):
    print("Reading Training Transforms")    
        
    train_light = args.train_light_folder

    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", train_light, white_background, extension, no_bg=no_bg)
    print(f"Read Train Transforms with {len(train_cam_infos)} cameras")

    if os.path.exists(os.path.join(path, "transforms_test.json")):
        test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", train_light, white_background, extension, no_bg=no_bg)
    else:
        test_cam_infos = []

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = test_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos)
    # nerf_normalization = {'translation': np.zeros([3], dtype=np.float32), 'radius': 1.}

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        if os.path.exists(os.path.join(path, 'rgbd')):
            import liblzfse  # https://pypi.org/project/pyliblzfse/
            def load_depth(filepath):
                with open(filepath, 'rb') as depth_fh:
                    raw_bytes = depth_fh.read()
                    decompressed_bytes = liblzfse.decompress(raw_bytes)
                    depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)
                    depth_img = depth_img.reshape((256, 192))
                return depth_img
            from utils.camera_utils import loadCam
            from collections import namedtuple
            ARGS = namedtuple('ARGS', ['resolution', 'data_device', 'load2gpu_on_the_fly'])
            args = ARGS(1, 'cpu', True)
            viewpoint_camera = loadCam(args, id, train_cam_infos[0], 1, [])
            w, h = viewpoint_camera.image_width, viewpoint_camera.image_height
            gt_depth = torch.from_numpy(load_depth(os.path.join(path, 'rgbd', '0.depth')))
            gt_depth = torch.nn.functional.interpolate(gt_depth[None, None], (h, w))[0, 0]
            far, near = viewpoint_camera.zfar, viewpoint_camera.znear
            u, v = torch.meshgrid(torch.linspace(.5, w-.5, w, device=gt_depth.device) / w * 2 - 1, torch.linspace(.5, h-.5, h, device = gt_depth.device) / h * 2 - 1, indexing='xy')
            u, v = u.reshape([-1]), v.reshape([-1])
            d = gt_depth.reshape([-1])
            nan_mask = d.isnan()
            nan_mask = torch.logical_or(nan_mask, d > 4)
            z = far / (far - near) * d - far * near / (far - near)
            uvz = torch.stack([u * d, v * d, z, d], dim=-1)
            pcl = uvz @ torch.inverse(viewpoint_camera.full_proj_transform)
            pcl = pcl[:, :3][~nan_mask]
            shs = torch.rand_like(pcl) / 255.0
            num_pts = shs.shape[0]
            pcd = BasicPointCloud(points=pcl, colors=SH2RGB(
                shs), normals=np.zeros((num_pts, 3)))
            xyz = pcl
        else:
            # Since this data set has no colmap data, we start with random points
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")

            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            # xyz = np.random.random((num_pts, 3)) * 20 - 10
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
                shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



############
def readColmapEnerfCameras(cam_extrinsics, cam_intrinsics, images_folder, msk_folder=None, timestep = 0, num_timesteps=0, start_timestep=0):
    cam_infos = []

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = f"cam{intr.id}_ts{timestep}"
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE" or intr.model == "OPENCV" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        if msk_folder is not None and image.mode == "RGB":
            msk_path = os.path.join(msk_folder, os.path.basename(extr.name))
            mask = Image.open(msk_path).convert("L")
            mask = mask.resize(image.size, Image.NEAREST)
            img_np = np.asarray(image).astype(np.float32)
            mask_np = np.asarray(mask).astype(np.float32) / 255.0
            mask_np = mask_np[..., None]
            bg = np.zeros_like(img_np)
            img_np = img_np * mask_np + bg * (1.0 - mask_np)
            img_np = img_np.astype(np.uint8)
            image = np.concatenate([img_np, (mask_np * 255).astype(np.uint8)], axis=-1)
            image = Image.fromarray(image)

        fid = int(timestep-start_timestep) / (num_timesteps - 1)
        
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, 
                            image_train_light=image, image_path_train_light=image_path, image_name_train_light=image_name, 
                            width=image.size[0], height=image.size[1], fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapEnerfSceneInfo(path, images, eval, apply_cam_norm=False, recenter_by_pcl=False):
    

    # --- Auto-detect sparse folder ---
    sparse_candidates = glob(os.path.join(path, "colmap*undistorted/sparse"))
    if not sparse_candidates:
        raise FileNotFoundError("No sparse folder found under path")
    sparse_name = os.path.relpath(sparse_candidates[0], path)

    # --- Auto-detect images folder ---
    image_candidates = glob(os.path.join(path, "images*undistorted"))
    image_dirs = [d for d in image_candidates if os.path.isdir(d)]
    if not image_dirs:
        raise FileNotFoundError("No images folder found under path")
    images = os.path.relpath(image_dirs[0], path)

    masks_candidates = glob(os.path.join(path, "masks*undistorted"))
    masks_dirs = [d for d in masks_candidates if os.path.isdir(d)]
    if not masks_dirs:
        raise FileNotFoundError("No masks folder found under path")
    masks = os.path.relpath(masks_dirs[0], path)

    print(f"[INFO] Using sparse folder: {sparse_name}")
    print(f"[INFO] Using images folder: {images}")
    print(f"[INFO] Using masks folder: {masks}")

    reading_dir = images
    reading_msk_dir = masks
    

    try:
        cameras_extrinsic_file = os.path.join(path, f"{sparse_name}", "images.bin")
        cameras_intrinsic_file = os.path.join(path, f"{sparse_name}", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, f"{sparse_name}", "images.txt")
        cameras_intrinsic_file = os.path.join(path, f"{sparse_name}", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)


    all_folders = sorted([f for f in os.listdir(os.path.join(path, reading_dir)) if os.path.isdir(os.path.join(path, reading_dir, f))])
    num_timesteps=len(all_folders)
    # all_folders = [f for f in all_folders if "740" in f] # quick way to train on one timestep only
    train_cam_infos = []
    test_cam_infos = []

    for folder in tqdm(all_folders, desc="Timestep"):
        cam_infos_unsorted = readColmapEnerfCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                                    images_folder=os.path.join(path, reading_dir, folder, "images"),
                                                    msk_folder=os.path.join(path, reading_msk_dir, folder, "masks"),
                                                    timestep=int(folder), num_timesteps=num_timesteps, start_timestep=int(all_folders[0]))
        cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name_train_light)

        
        
        eval_cam = "16"
        if eval:
            train_cam_infos.extend([c for idx, c in enumerate(
                cam_infos) if eval_cam not in c.image_name_train_light])
            test_cam_infos.extend([c for idx, c in enumerate(
                cam_infos) if eval_cam in c.image_name_train_light])
        else:
            train_cam_infos.extend(cam_infos)

    nerf_normalization = getNerfppNorm(train_cam_infos + test_cam_infos, apply=apply_cam_norm)

    if recenter_by_pcl:
        ply_path = os.path.join(path, f"points3d_recentered.ply")
    elif apply_cam_norm:
        ply_path = os.path.join(path, f"points3d_normalized.ply")
    else:
        ply_path = os.path.join(path, f"points3d.ply")

    merged_path = os.path.join(path, "merged_pcd.ply")
    bin_path = os.path.join(path, f"{sparse_name}/points3D.bin")
    txt_path = os.path.join(path, f"{sparse_name}/points3D.txt")
    adj_path = os.path.join(path, f"{sparse_name}/camera_adjustment")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        
        if os.path.exists(merged_path):
            xyz, rgb, _ = read_ply_to_numpy(merged_path)
            print("merged_pcd.ply found!")
        else:
            raise FileNotFoundError(
                "merged_pcd.ply not found! Make sure your point cloud has dense points on the actor.\n"
                "See notebooks/enerf_prepare_colmap_masks.ipynb for guidelines."
            )
            # try:
            #     xyz, rgb, _ = read_points3D_binary(bin_path)
            # except:
            #     xyz, rgb, _ = read_points3D_text(txt_path)
        if apply_cam_norm:
            xyz += nerf_normalization["apply_translate"]
            xyz /= nerf_normalization["apply_radius"]
        if recenter_by_pcl:
            pcl_center = xyz.mean(axis=0)
            translate_cam_info(train_cam_infos, - pcl_center)
            translate_cam_info(test_cam_infos, - pcl_center)
            xyz -= pcl_center
            np.savez(adj_path, translate=-pcl_center)
        storePly(ply_path, xyz, rgb)
    elif recenter_by_pcl:
        translate = np.load(adj_path + '.npz')['translate']
        translate_cam_info(train_cam_infos, translate=translate)
        translate_cam_info(test_cam_infos, translate=translate)
    
    try:
        pcd = fetchPly(ply_path)
    except:
        try:
            pcd = fetchPly_from_gaussians(ply_path)
        except:
            pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           all_timesteps = sorted(all_folders))
    return scene_info

def read_ply_to_numpy(ply_path):
    """
    Reads a PLY file and returns xyz and rgb arrays.

    Returns:
        xyz (Nx3 np.ndarray): 3D coordinates
        rgb (Nx3 np.ndarray): RGB values in uint8
        dummy (None): placeholder to match (xyz, rgb, _) pattern
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(pcd.points)
    rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    if rgb.size == 0 or rgb.shape[0] == 0:
        print("[WARNING] RGB data is empty, filling with default gray color (0.5, 0.5, 0.5)")
        rgb = np.ones_like(xyz) * 0.5  # float32 in [0,1]
    
    rgb = (rgb * 255).astype(np.uint8)
    print("XYZ shape:", xyz.shape)
    print("RGB shape:", rgb.shape)
    return xyz, rgb, None


def readCamerasDNARendering(path, info_dict, white_background, image_scaling=0.5, return_smplx=False, start_frame=0, end_frame=-1):
    output_view = info_dict["views"]
    ratio = image_scaling

    cam_infos = []
    main_file = path
    annot_file = path.replace('main', 'annotations').split('.')[0] + '_annots.smc'
    main_reader = SMCReader(main_file)
    annot_reader = SMCReader(annot_file)
    
    start_timestep = start_frame
    end_timestep =  main_reader.get_Camera_5mp_info()["num_frame"]-1 if end_frame==-1 else end_frame
    

    actual_start_timestep = max(start_timestep, 0)
    actual_end_timestep = min(end_timestep, main_reader.get_Camera_5mp_info()["num_frame"])
    num_timesteps = actual_end_timestep - actual_start_timestep
    print("DNA SCENE: Sanity check of start, end timsetep:", actual_start_timestep, actual_end_timestep)

    for frame_idx in tqdm(range(actual_start_timestep, actual_end_timestep, 1), desc="Processing frames", ncols=100):
        if frame_idx==start_timestep:
            smplx_vertices = None
            if return_smplx:
                gender = main_reader.actor_info['gender']
                model = SMPLX(
                    'assets/body_models/smplx/', smpl_type='smplx',
                    gender=gender, use_face_contour=True, flat_hand_mean=False, use_pca=False,
                    num_betas=10, num_expression_coeffs=10, ext='npz'
                )
                smplx_dict = annot_reader.get_SMPLx(Frame_id=frame_idx)
                betas = torch.from_numpy(smplx_dict["betas"]).unsqueeze(0).float()
                expression = torch.from_numpy(smplx_dict["expression"]).unsqueeze(0).float()
                fullpose = torch.from_numpy(smplx_dict["fullpose"]).unsqueeze(0).float()
                translation = torch.from_numpy(smplx_dict['transl']).unsqueeze(0).float()
                output = model(
                    betas=betas,
                    expression=expression,
                    global_orient=fullpose[:, 0].clone(),
                    body_pose=fullpose[:, 1:22].clone(),
                    jaw_pose=fullpose[:, 22].clone(),
                    leye_pose=fullpose[:, 23].clone(),
                    reye_pose=fullpose[:, 24].clone(),
                    left_hand_pose=fullpose[:, 25:40].clone(),
                    right_hand_pose=fullpose[:, 40:55].clone(),
                    transl=translation,
                    return_verts=True)
                smplx_vertices = output.vertices.detach().cpu().numpy().squeeze()

        parent_dir = os.path.dirname(os.path.dirname(path))
        out_img_dir = os.path.join(parent_dir, "images")
        # os.makedirs(out_img_dir, exist_ok=True)
        bg = np.array([255, 255, 255]) if white_background else np.array([0, 0, 0])
        idx = 0
        for view_index in output_view:
            # Load K, R, T
            cam_params = annot_reader.get_Calibration(view_index)
            K = cam_params['K']
            D = cam_params['D']  # k1, k2, p1, p2, k3
            RT = cam_params['RT']

            # Load image, mask
            image = main_reader.get_img('Camera_5mp', view_index, Image_type='color', Frame_id=frame_idx)
            image = cv.undistort(image, K, D)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            mask = annot_reader.get_mask(view_index, Frame_id=frame_idx)
            mask = cv.undistort(mask, K, D)
            mask = mask[..., np.newaxis].astype(np.float32) / 255.0
            image = image * mask + bg * (1.0 - mask)

            c2w = np.array(RT, dtype=np.float32)
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3].copy()
            if ratio != 1.0:
                H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
                image = cv.resize(image, (W, H), interpolation=cv.INTER_AREA)
                mask = cv.resize(mask, (W, H), interpolation=cv.INTER_NEAREST)
                K[:2] = K[:2] * ratio

            H, W, _ = image.shape
            focalX = K[0, 0]
            focalY = K[1, 1]
            FovX = focal2fov(focalX, W)
            FovY = focal2fov(focalY, H)

            # image = Image.fromarray(np.array(image, dtype=np.byte), "RGB")
            #our change: we need this mask as last channel
            image = np.concatenate([image, np.expand_dims(mask*255, -1)], axis=-1)
            image = Image.fromarray(np.array(image, dtype=np.uint8))

            image_name = "view%04d_frame%04d" % (view_index, frame_idx)
            image_path = os.path.join(out_img_dir, "%s.png" % image_name)

            fid = 0 if num_timesteps <=1 else int(frame_idx - start_timestep) / (num_timesteps - 1)

            zfar = 100.0
            znear = 0.01
            proj_matrix_DNA = torch.tensor(getProjectionMatrix_DNA(znear, zfar, K, W, H).transpose(0,1))

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                                        image_train_light=image, image_path_train_light=image_path,
                                        image_name_train_light=image_name,
                                        width=image.size[0], height=image.size[1], fid=fid, proj_matrix=proj_matrix_DNA))

            idx += 1

    return cam_infos, smplx_vertices


def readDNARenderingInfo(path, white_background, eval, load_test_set_only, start_frame = 0, end_frame =-1):
    test_view_arr = [3, 5, 7, 8, 11, 15, 17, 19, 21, 23, 27, 29, 31, 33, 35, 39, 41, 45]
    all_train_candidates = [x for x in range(48) if x not in test_view_arr]
    # train_view_arr = random.sample(all_train_candidates, 20)
    train_view_arr = all_train_candidates
    test_view_arr = [21] #test_view_arr[-5:-6] #for table 31, for shoes 21

    train_info_dict = {
        "views": train_view_arr,
        "frame_idx": 1,
    }
    test_info_dict = {
        "views": test_view_arr,
        "frame_idx": 1,
    }
    print("Reading Training Transforms", flush=True)

    if load_test_set_only:
        print("TRAIN CAMS ALL BUT ONE SKIPPED, LOAD TEST SET ONLY ")

        train_info_dict = {
        "views": train_view_arr[:1],
        "frame_idx": 1,
        }
        train_cam_infos, smplx_vertices = readCamerasDNARendering(path, train_info_dict, white_background,
                                                                return_smplx=True, start_frame=start_frame, end_frame=end_frame)
    else:
        train_cam_infos, smplx_vertices = readCamerasDNARendering(path, train_info_dict, white_background,
                                                                return_smplx=True, start_frame=start_frame, end_frame=end_frame)
    print("Reading Test Transforms", flush=True)
    test_cam_infos, _ = readCamerasDNARendering(path, test_info_dict, white_background, return_smplx=False,start_frame=start_frame, end_frame=end_frame)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    parent_dir = os.path.dirname(os.path.dirname(path))
    ply_path = os.path.join(parent_dir, "points3d.ply")
    print("Using SMPLX vertices to initiate", flush=True)
    num_pts, _ = smplx_vertices.shape

    # --- Add some more points, there are only about 10k vertices from smplx  ---
    bbox_min = smplx_vertices.min(axis=0)
    bbox_max = smplx_vertices.max(axis=0)
    scene_scale = np.linalg.norm(bbox_max - bbox_min)
    osc_amp = 0.01 * scene_scale  # 0.8% of total scene extent
    n = 8
    perturbations = [smplx_vertices + np.random.normal(scale=osc_amp, size=smplx_vertices.shape) for _ in range(n)]
    augmented_vertices = np.vstack([smplx_vertices] + perturbations)
    smplx_vertices = augmented_vertices
    num_pts, _ = smplx_vertices.shape

    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=smplx_vertices, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    storePly(ply_path, smplx_vertices, SH2RGB(shs) * 255)

    # if point cloud from some gaussian training exits, override the pcd
    point_cloud_from_gaussians_path = os.path.join(parent_dir, "point_cloud.ply")
    if os.path.exists(point_cloud_from_gaussians_path):
        print("$$$ DNA SCENE $$$ USES GAUSSIAN PLY AS INIT INSTEAD OF SMPLX")
        pcd = fetchPly_from_gaussians(point_cloud_from_gaussians_path)
        
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "ColmapENerf":readColmapEnerfSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "DNA-Rendering": readDNARenderingInfo,
}
