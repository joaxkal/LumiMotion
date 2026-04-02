import torch
import torch.nn.functional as F
import imageio
import numpy as np
from . import renderutils as ru
from utils.light_utils import *
import nvdiffrast.torch as dr
import pyexr
from utils.graphics_utils import srgb_to_rgb, rgb_to_srgb
import cv2

def inverse_sigmoid(x):
    return torch.log(x/(1-x))


class EnvLightMip(torch.nn.Module):

    def __init__(self, path=None, device=None, scale=1.0, min_res=16, max_res=128, min_roughness=0.08, max_roughness=0.5):
        super().__init__()
        self.device = device if device is not None else 'cuda' # only supports cuda
        self.scale = scale # scale of the hdr values
        self.min_res = min_res # minimum resolution for mip-map
        self.max_res = max_res # maximum resolution for mip-map
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness

        # init an empty cubemap
        self.base = torch.nn.Parameter(
            torch.zeros(6, self.max_res, self.max_res, 3, dtype=torch.float32, device=self.device),
            requires_grad=True,
        )
        
        self.transform = None
        # try to load from file (.hdr or .exr)
        if path is not None:
            self.load(path)
        
        self.build_mips()

    def set_transform(self, transform):
        self.transform = transform

    def load(self, path):
        """
        Load an .hdr or .exr environment light map file and convert it to cubemap.
        """
        hdr_image = imageio.imread(path)
        
        if hdr_image.dtype != np.float32:
            raise ValueError("HDR image should be in float32 format.")

        ldr_image = rgb_to_srgb(hdr_image)
        image = torch.from_numpy(ldr_image).to(self.device) *  self.scale
        image = torch.clamp(image, 0.001 , 1-0.001)
        image = inverse_sigmoid(image)

        # Convert from latlong to cubemap format
        cubemap = latlong_to_cubemap(image, [self.max_res, self.max_res], self.device)

        # Assign the cubemap to the base parameter
        self.base.data = cubemap 

    def build_mips(self, cutoff=0.99):
        """
        Build mip-maps for specular reflection based on cubemap.
        """
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.min_res:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.max_roughness - self.min_roughness) + self.min_roughness
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 

        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def get_mip(self, roughness):
        """
        Map roughness to mip level.
        """
        return torch.where(
            roughness < self.max_roughness, 
            (torch.clamp(roughness, self.min_roughness, self.max_roughness) - self.min_roughness) / (self.max_roughness - self.min_roughness) * (len(self.specular) - 2), 
            (torch.clamp(roughness, self.max_roughness, 1.0) - self.max_roughness) / (1.0 - self.max_roughness) + len(self.specular) - 2
        )
        

    def __call__(self, l, mode=None, roughness=None):
        """
        Query the environment light based on direction and roughness.
        """
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # Reshape to [B, H, W, -1] if necessary
            l = l.reshape(1, 1, -1, l.shape[-1])
            if self.transform is not None:
                l = l @ self.transform.T
            if roughness is not None:
                roughness = roughness.reshape(1, 1, -1, 1)

        if mode == "diffuse":
            # Diffuse lighting
            light = dr.texture(self.diffuse[None, ...], l, filter_mode='linear', boundary_mode='cube')
        elif mode == "pure_env":
            # Pure environment light (no mip-map)
            light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        else:
            # Specular lighting with mip-mapping
            miplevel = self.get_mip(roughness)
            light = dr.texture(
                self.specular[0][None, ...], 
                l,
                mip=list(m[None, ...] for m in self.specular[1:]), 
                mip_level_bias=miplevel[..., 0], 
                filter_mode='linear-mipmap-linear', 
                boundary_mode='cube'
            )

        light = light.view(*prefix, -1)
        
        return torch.sigmoid(light)
    
def pixel_grid(width, height, center_x = 0.5, center_y = 0.5):
    y, x = torch.meshgrid(
            (torch.arange(0, height, dtype=torch.float32, device="cuda") + center_y) / height, 
            (torch.arange(0, width, dtype=torch.float32, device="cuda") + center_x) / width)
    return torch.stack((x, y), dim=-1)


def get_env_direction1(H, W):
    gy, gx = torch.meshgrid(torch.linspace(0.0 + 0.5 / H, 1.0 - 0.5 / H, H, device='cuda'),
                            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device='cuda'),
                            indexing='ij')
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    env_directions = torch.stack((
        sintheta*sinphi,
        costheta,
        -sintheta*cosphi
        ), dim=-1)
    return env_directions


def get_env_direction2(H, W):
    gy, gx = torch.meshgrid(torch.linspace(0.0 + 0.5 / H, 1.0 - 0.5 / H, H, device='cuda'),
                            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device='cuda'),
                            indexing='ij')
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    env_directions = torch.stack((
        sintheta*cosphi,
        -sintheta*sinphi,
        costheta,
        ), dim=-1)
    return env_directions


class EnvLight(torch.nn.Module):
    def __init__(self, path=None, device=None, resolution=None, min_res=8, max_res=128, min_roughness=0.08, max_roughness=0.5, activation='exp', init_value=0.5):
        super().__init__()
        self.device = device if device is not None else 'cuda' # only supports cuda
        self.min_res = min_res # minimum resolution for mip-map
        self.max_res = max_res # maximum resolution for mip-map
        self.resolution = resolution
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness
        self.optimizer = None

        # for map render
        self.env_H, self.env_W = 256, 512

        if path is not None:
            latlong_img = self.load(path)
            #print('Loaded ', path)
            if resolution is None:
                resolution = latlong_img.shape[:2]
            texcoord = pixel_grid(resolution[1], resolution[0])
            latlong_img = dr.texture(latlong_img[None, ...], texcoord[None, ...], filter_mode='linear')[0].clamp_min(1e-4)
            self.base = torch.nn.Parameter(latlong_img, requires_grad=True)
        else:
            self.base = torch.nn.Parameter(
                torch.full((resolution[0], resolution[1], 3), init_value, dtype=torch.float32, device=self.device),
                requires_grad=True,
            )
        self.transform = None
        self.base_mip = None
        
        # self.build_mips()
        self.activation_name = activation
        if activation == 'sigmoid':
            self.activation = torch.sigmoid
            self.base.data[:] = inverse_sigmoid(self.base.data)
        elif activation == 'exp':
            self.activation = torch.exp
            self.base.data[:] = torch.log(self.base.data)
        elif activation == 'none':
            self.activation = lambda x: x
        else:
            raise NotImplementedError

    def training_setup(self, training_args):
        l = [{'params': self.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env"},]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def update_pdf(self):
        with torch.no_grad():
            # Compute PDF
            Y = pixel_grid(self.base.shape[1], self.base.shape[0])[..., 1]
            self._pdf = torch.max(self.activation(self.base).clamp_min(0.0), dim=-1)[0] * torch.sin(Y * np.pi) # Scale by sin(theta) for lat-long, https://cs184.eecs.berkeley.edu/sp18/article/25
            self._pdf = self._pdf / torch.sum(self._pdf)

    def sample_light_directions(self, B, sample_num, training=False):
        pdf_flat = self._pdf.reshape(-1)
        light_dir_idx = torch.multinomial(pdf_flat, B*sample_num, replacement=True)
        
        H, W = self._pdf.shape[:2]
        gx = ((light_dir_idx % W  + 0.5) / W) * 2 - 1
        gy = (light_dir_idx // W + 0.5) / H
        if training:
            gx = gx + (torch.rand_like(gx) - 0.5) / W * 2
            gy = gy + (torch.rand_like(gy) - 0.5) / H
        sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
        sinphi, cosphi = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
        direction = torch.stack((
            sintheta*sinphi, 
            costheta, 
            -sintheta*cosphi
        ), dim=-1)
        
        if self.transform is not None:
            direction = direction @ self.transform
        direction = direction.reshape(B, sample_num, 3)
        
        probability = self.light_pdf(direction)
        
        return direction, probability

    def light_pdf(self, direction):
        pdf_flat = self._pdf.reshape(-1)
        direction_flat = direction.reshape(-1, 3)
        if self.transform is not None:
            direction_flat = direction_flat @ self.transform.T
        H, W = self._pdf.shape[:2]
        
        u = (torch.atan2(direction_flat[..., 0], -direction_flat[..., 2]).nan_to_num() / (2.0 * torch.pi) + 0.5)
        v = torch.acos(direction_flat[..., 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / torch.pi
        
        u_idx = (u * W).clamp(0, W - 1).long()
        v_idx = (v * H).clamp(0, H - 1).long()
        light_dir_idx = u_idx + v_idx * W
        
        pdf_weight = H * W / (2.0 * torch.pi ** 2 * torch.sin(v * torch.pi).clamp_min(1e-6))
        probability = (torch.take_along_dim(pdf_flat, light_dir_idx, dim=0) * pdf_weight).reshape(*direction.shape[:2], 1)
        return probability

    def capture(self):
        state_dict = super().state_dict()
        return {
            "state_dict": state_dict,
            "activation": self.activation_name
        }

    def render_env_map(self, H=128):
        if H == self.env_H:
            directions1 = self.env_directions1
            directions2 = self.env_directions2
        else:
            W = H * 2
            directions1 = get_env_direction1(H, W)
            directions2 = get_env_direction2(H, W)
        return {'env1': self(directions1, mode="pure_env"), 'env2': self(directions2, mode="pure_env")}


    def load_weights(self, weights_path, iteration):
        out_weights_path = os.path.join(weights_path, "envmap/iteration_{}".format(iteration), 'envmap.pth')
        model_args = torch.load(out_weights_path, map_location='cpu')
        self.restore(model_args)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "envmap/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        env_dict = self.render_env_map()

        grid = [
            env_dict["env1"].permute(2, 0, 1),
            env_dict["env2"].permute(2, 0, 1),
        ]
        hdr_tensor = grid[1]  # C, H, W
        hdr_np = hdr_tensor.permute(1, 2, 0).cpu().numpy().astype(np.float32)  # H, W, C

        # Save output HDR
        cv2.imwrite(os.path.join(out_weights_path, 'envmap.hdr'), cv2.cvtColor(hdr_np, cv2.COLOR_RGB2BGR))
        #print('Saved envmap ', os.path.join(out_weights_path, 'envmap.hdr'))
        torch.save(self.capture(), os.path.join(out_weights_path, 'envmap.pth'))

    def restore(self, model_args):
        activation = model_args['activation']
        self.activation_name = activation
        if activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'exp':
            self.activation = torch.exp
        elif activation == 'none':
            self.activation = lambda x: x
        else:
            raise NotImplementedError
        self.load_state_dict(model_args['state_dict'])
    
    def set_transform(self, transform):
        self.transform = transform

    def load(self, path):
        """
        Load an .hdr or .exr environment light map file and convert it to cubemap.
        """
        if path.endswith(".exr"):
            image = pyexr.open(path).get()[:, :, :3]
        elif path.endswith(".hdr"):  # #
            #image = imageio.imread(path, format='HDR-FI')[:, :, :3]
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = srgb_to_rgb(imageio.imread(path)[:, :, :3] / 255)
            
        image = torch.from_numpy(image).to(self.device)
        return image



    def build_mips(self, cutoff=0.99):
        """
        Build mip-maps for specular reflection based on cubemap.
        """
        self.base_mip = latlong_to_cubemap(self.base, [self.max_res, self.max_res], self.device)
        
        self.specular = [self.base_mip]
        while self.specular[-1].shape[1] > self.min_res:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.max_roughness - self.min_roughness) + self.min_roughness
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 

        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def get_mip(self, roughness):
        """
        Map roughness to mip level.
        """
        return torch.where(
            roughness < self.max_roughness, 
            (torch.clamp(roughness, self.min_roughness, self.max_roughness) - self.min_roughness) / (self.max_roughness - self.min_roughness) * (len(self.specular) - 2), 
            (torch.clamp(roughness, self.max_roughness, 1.0) - self.max_roughness) / (1.0 - self.max_roughness) + len(self.specular) - 2
        )

    def __call__(self, l, mode='pure_env', roughness=None):
        """
        Query the environment light based on direction and roughness.
        """
                
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # Reshape to [B, H, W, -1] if necessary
            l = l.reshape(1, 1, -1, l.shape[-1])
            if self.transform is not None:
                l = l @ self.transform.T
            if roughness is not None:
                roughness = roughness.reshape(1, 1, -1, 1)

        if mode == "diffuse":
            # Diffuse lighting
            light = dr.texture(self.diffuse[None, ...], l.contiguous(), filter_mode='linear', boundary_mode='cube')
        elif mode == "pure_env":
            uv = torch.cat([
                (torch.atan2(l[..., :1], -l[..., 2:3]).nan_to_num() / (2.0 * torch.pi) + 0.5),
                torch.acos(l[..., 1:2].clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / torch.pi,
            ], dim=-1).clamp(0, 1)
            light = dr.texture(self.base[None, ...], uv, filter_mode='linear')
        else:
            # Specular lighting with mip-mapping
            miplevel = self.get_mip(roughness)
            light = dr.texture(
                self.specular[0][None, ...], 
                l,
                mip=list(m[None, ...] for m in self.specular[1:]), 
                mip_level_bias=miplevel[..., 0], 
                filter_mode='linear-mipmap-linear', 
                boundary_mode='cube'
            )
        light = light.view(*prefix, -1)
        
        # return self.activation(light).clamp(0, 3)
        return self.activation(light).clamp_min(0.0)


class EnvMap(torch.nn.Module):
    def __init__(self, path=None, scale=1.0):
        super().__init__()
        self.device = "cuda"  # only supports cuda
        self.scale = scale  # scale of the hdr values
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")

        self.envmap = self.load(path, scale=self.scale, device=self.device)
        self.transform = None

    @staticmethod
    def load(envmap_path, scale, device):
        if not envmap_path.endswith(".exr"):
            image = srgb_to_rgb(imageio.imread(envmap_path)[:, :, :3] / 255)
        elif envmap_path.endswith(".hdr"):  # #
            # Load HDR file
            image = imageio.imread(envmap_path, format='HDR-FI')[:, :, :3]
        else:
            # load latlong env map from file
            image = pyexr.open(envmap_path).get()[:, :, :3]

        image = image * scale

        env_map_torch = torch.tensor(image, dtype=torch.float32, device=device, requires_grad=False)

        return env_map_torch

    def __call__(self, dirs, mode='pure_env', roughness=None, transform=None):
        shape = dirs.shape
        dirs = dirs.reshape(-1, 3)

        if transform is not None:
            dirs = dirs @ transform.T
        elif self.transform is not None:
            dirs = dirs @ self.transform.T

        envir_map =  self.envmap.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
        phi = torch.arccos(dirs[:, 2]).reshape(-1) - 1e-6
        theta = torch.atan2(dirs[:, 1], dirs[:, 0]).reshape(-1)
        # normalize to [-1, 1]
        query_y = (phi / np.pi) * 2 - 1
        query_x = - theta / np.pi
        grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)
        light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
    
        return light_rgbs.reshape(*shape)


class DirectLightMap(torch.nn.Module):

    def __init__(self, max_res=16, init_value=0.5, **kwargs):
        super().__init__()
        self.H = max_res
        self.W = max_res * 2
        env = (init_value * torch.rand((1, self.H, self.W, 3), device="cuda"))
        self.env = torch.nn.Parameter(env, requires_grad=True)
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")
        
    def __call__(self, dirs, mode='pure_env', roughness=None, transform=None):
        dirs = F.normalize(dirs.detach(), dim=-1)
        shape = dirs.shape
        dirs = dirs.reshape(-1, 3)
        envir_map = self.get_env.permute(0, 3, 1, 2) # [1, 3, H, W]
        phi = torch.arccos(dirs[:, 2]).reshape(-1) - 1e-6
        theta = torch.atan2(dirs[:, 1], dirs[:, 0]).reshape(-1)
        # normalize to [-1, 1]
        query_y = (phi / np.pi) * 2 - 1
        query_x = - theta / np.pi
        grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)
        light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
        return F.softplus(light_rgbs).reshape(*shape)

    @property
    def get_env(self):
        return F.softplus(self.env)
    
