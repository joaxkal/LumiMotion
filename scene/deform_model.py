import torch
from utils.time_utils import DeformNetwork, StaticNetwork
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


model_dict = {'mlp': DeformNetwork, 'static': StaticNetwork}


class DeformModel:
    def __init__(self, deform_type='mlp', is_blender=False, **kwargs):
        if deform_type not in model_dict:
            raise ValueError(f"Unsupported deform_type '{deform_type}'. Supported: {list(model_dict.keys())}")
        self.deform = model_dict[deform_type](is_blender=is_blender, **kwargs).cuda()
        self.name = self.deform.name
        self.optimizer = None
        self.spatial_lr_scale = 5

    @property
    def reg_loss(self):
        return self.deform.reg_loss

    def step(self, xyz, time_emb, iteration=0, **kwargs):
        return self.deform(xyz, time_emb, iteration=iteration, **kwargs)

    def train_setting(self, training_args):
        l = [
            {'params': group['params'],
             'lr': training_args.position_lr_init * self.spatial_lr_scale * training_args.deform_lr_scale,
             "name": group['name']}
             for group in self.deform.trainable_parameters()
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale * training_args.deform_lr_scale, lr_final=training_args.position_lr_final * training_args.deform_lr_scale, lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        if os.path.exists(weights_path):
            self.deform.load_state_dict(torch.load(weights_path))
            return True
        else:
            return False

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform" or param_group["name"] == "mlp" or \
                param_group["name"] == "mlp_other_params" or param_group["name"] == "mlp_color": #after update of naming
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
        
    def update(self, iteration):
        self.deform.update(iteration)
