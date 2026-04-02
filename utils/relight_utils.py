import torch
import torch.nn.functional as F
import numpy as np


def fibonacci_sphere_sampling(normals, sample_num, random_rotate=True):
    pre_shape = normals.shape[:-1]
    if len(pre_shape) > 1:
        normals = normals.reshape(-1, 3)
    delta = np.pi * (3.0 - np.sqrt(5.0))

    # fibonacci sphere sample around z axis
    idx = torch.arange(sample_num, dtype=torch.float, device='cuda')[None]
    z = (1 - 2 * idx / (2 * sample_num - 1))#.clamp_min(np.sin(10/180*np.pi))
    rad = torch.sqrt(1 - z ** 2)
    theta = delta * idx
    if random_rotate:
        theta = torch.rand(*pre_shape, 1, device='cuda') * 2 * np.pi + theta
    y = torch.cos(theta) * rad
    x = torch.sin(theta) * rad
    z_samples = torch.stack([x, y, z.expand_as(y)], dim=-2)

    # rotate to normal
    # z_vector = torch.zeros_like(normals)
    # z_vector[..., 2] = 1  # [H, W, 3]
    # rotation_matrix = rotation_between_vectors(z_vector, normals)
    rotation_matrix = rotation_between_z(normals)
    incident_dirs = rotation_matrix @ z_samples
    incident_dirs = F.normalize(incident_dirs, dim=-2).transpose(-1, -2)
    incident_areas = torch.ones_like(incident_dirs)[..., 0:1] * 2 * np.pi
    if len(pre_shape) > 1:
        incident_dirs = incident_dirs.reshape(*pre_shape, sample_num, 3)
        incident_areas = incident_areas.reshape(*pre_shape, sample_num, 1)
    return incident_dirs, incident_areas

def sample_incident_rays(normals, is_training=False, sample_num=24):
    if is_training:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=True)
    else:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=False)

    return incident_dirs, incident_areas  # [N, S, 3], [N, S, 1]


def rotation_between_z(vec):
    """
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    Args:
        vec: [..., 3]

    Returns:
        R: [..., 3, 3]

    """
    v1 = -vec[..., 1]
    v2 = vec[..., 0]
    v3 = torch.zeros_like(v1)
    v11 = v1 * v1
    v22 = v2 * v2
    v33 = v3 * v3
    v12 = v1 * v2
    v13 = v1 * v3
    v23 = v2 * v3
    cos_p_1 = (vec[..., 2] + 1).clamp_min(1e-7)
    R = torch.zeros(vec.shape[:-1] + (3, 3,), dtype=torch.float32, device="cuda")
    R[..., 0, 0] = 1 + (-v33 - v22) / cos_p_1
    R[..., 0, 1] = -v3 + v12 / cos_p_1
    R[..., 0, 2] = v2 + v13 / cos_p_1
    R[..., 1, 0] = v3 + v12 / cos_p_1
    R[..., 1, 1] = 1 + (-v33 - v11) / cos_p_1
    R[..., 1, 2] = -v1 + v23 / cos_p_1
    R[..., 2, 0] = -v2 + v13 / cos_p_1
    R[..., 2, 1] = v1 + v23 / cos_p_1
    R[..., 2, 2] = 1 + (-v22 - v11) / cos_p_1
    R = torch.where((vec[..., 2] + 1 > 0)[..., None, None], R,
                    -torch.eye(3, dtype=torch.float32, device="cuda").expand_as(R))
    return R
