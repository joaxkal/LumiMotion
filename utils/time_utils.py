import torch
import torch.nn as nn
import torch.nn.functional as F


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        self.N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class StaticNetwork(nn.Module):
    def __init__(self, return_tensors=False, *args, **kwargs) -> None:
        super().__init__()
        self.name = 'static'
        self.param = nn.Parameter(torch.zeros([1]).cuda())
        self.reg_loss = 0.
        self.return_tensors = return_tensors

    def forward(self, x, t, **kwargs):
        if self.return_tensors:
            return_dict = {'d_xyz': torch.zeros_like(x), 'd_rotation': torch.zeros_like(x[..., [0, 0, 0, 0]]), 'd_scaling': torch.zeros_like(x), 'local_rotation': torch.zeros_like(x[..., [0, 0, 0, 0]]), 'hidden': None, 'd_opacity':None, 'd_color': None}
        else:
            return_dict = {'d_xyz': 0., 'd_rotation': 0., 'd_scaling': 0., 'hidden': 0., 'd_opacity':None, 'd_color': None}
        return return_dict
    
    def trainable_parameters(self):
        return [{'params': [self.param], 'name': 'deform'}]
    
    def update(self, *args, **kwargs):
        return


class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, t_multires=6, multires=10,
                 is_blender=False, pred_color=False, **kwargs):
        super(DeformNetwork, self).__init__()
        self.name = 'mlp'
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)

        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        
        self.input_ch = xyz_input_ch + time_input_ch

        self.pred_color = pred_color

        self.reg_loss = 0.

        if is_blender:
            # Better for D-NeRF Dataset
            self.time_out = 30

            self.timenet = nn.Sequential(
                nn.Linear(time_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out))

            self.linear = nn.ModuleList(
                [nn.Linear(xyz_input_ch + self.time_out, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.time_out, W)
                    for i in range(D - 1)]
            )

        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )

        self.is_blender = is_blender

        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_scaling = nn.Linear(W, 2)
        self.gaussian_rotation = nn.Linear(W, 4)

        for layer in self.linear:
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)

        nn.init.normal_(self.gaussian_warp.weight, mean=0, std=1e-5)
        nn.init.normal_(self.gaussian_scaling.weight, mean=0, std=1e-8)
        nn.init.normal_(self.gaussian_rotation.weight, mean=0, std=1e-5)
        nn.init.zeros_(self.gaussian_warp.bias)
        nn.init.zeros_(self.gaussian_scaling.bias)
        nn.init.zeros_(self.gaussian_rotation.bias)

        if self.pred_color:
            in_dim = self.linear[0].weight.shape[-1] + W
            self.gaussian_color = nn.Sequential(nn.Linear(in_dim, W), nn.ReLU(), 
                                                nn.Linear(W, W), nn.ReLU(), 
                                                nn.Linear(W, 6))
            print("################ gaussian_color params:", sum(p.numel() for p in self.gaussian_color.parameters()))
            
            for layer in self.gaussian_color:
                if hasattr(layer, 'weight'):
                    nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            nn.init.normal_(self.gaussian_color[-1].weight, mean=0, std=1e-5)
            
    
    def trainable_parameters(self):
        all_params = list(self.parameters())

        color_params = []
        if self.pred_color and hasattr(self, "gaussian_color"):
            color_params = list(self.gaussian_color.parameters())

        color_ids = {id(p) for p in color_params}
        other_params = [p for p in all_params if id(p) not in color_ids]

        groups = []
        if other_params:
            groups.append({'params': other_params, 'name': 'mlp_other_params'})
        if color_params:
            groups.append({'params': color_params, 'name': 'mlp_color'})
        return groups

    def forward(self, x, t, **kwargs):

        binary_features = kwargs["feature"]
        t_emb = self.embed_time_fn(t)
        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        return_dict = {'d_xyz': d_xyz*binary_features, 
                       'd_rotation': rotation*binary_features, 
                       'd_scaling': scaling*0, 
                       'hidden': h,
                       'd_opacity': None}
        if self.pred_color:
            delta_color = self.gaussian_color(torch.cat([x_emb, t_emb, h], dim=-1))
            # shadowed color: c_canonical*(1-c_shad)
            shadowed_color = delta_color[:, :3]
            
            # Dynamic color: c_canonical+c_dyn; finally not used
            dynamic_color = delta_color[:, 3:]*binary_features

            final_delta_color = torch.cat([shadowed_color, dynamic_color], dim=-1)
            return_dict['d_color'] = final_delta_color
        else:
            return_dict['d_color'] = None
        return return_dict
    
    def update(self, iteration, *args, **kwargs):
        return
