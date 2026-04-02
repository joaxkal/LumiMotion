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

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            # if shorthand:
            #     if t == bool:
            #         group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
            #     else:
            #         group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            # else:
            if t == bool:
                group.add_argument("--" + key, default=value, action="store_true")
            else:
                group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):

        
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self.train_light_folder = "chapel_day_4k_32x16_rot0"
        self.test_light_folder = "golden_bay_4k_32x16_rot330"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.load2gpu_on_the_fly = False
        self.is_blender = False
        self.deform_type = 'mlp'
        self.hyper_dim = 1 # its for static-dynamic learnable variable
        self.pred_color = True
        self.no_binary_separation = False
        self.load_test_set_only = False
        self.start_frame = 0 #Only used in DNA data
        self.end_frame = -1 #Only used in DNA data
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        if not g.model_path.endswith(g.deform_type):
            g.model_path = os.path.join(os.path.dirname(os.path.normpath(g.model_path)), os.path.basename(os.path.normpath(g.model_path)) + f'_{g.deform_type}')
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.depth_ratio = 1.0 #careful here, it can really breake relighting

        self.light_sample_num = 0 # we do not use it (its for importance sampling which is commented out right now)
        self.diffuse_sample_num = 256
        self.light_t_min = 0.1
        
        # Here options from IRGS ablation. We use full model, so all false.
        self.wo_indirect = False
        self.wo_indirect_relight = False
        self.detach_indirect = False
        self.wo_specular = False

        super().__init__(parser, "Pipeline Parameters")




class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 80_000
        self.warm_up = 1000 #1_000
        self.binarization_warm_up = 1000

        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.deform_lr_max_steps = 40_000
        self.feature_lr = 0.004 #feature is variable for static - dynamic separation #0.0025
        self.opacity_lr = 0.05
        self.roughness_lr = 0.002 #0.001
        self.scaling_lr = 0.002
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100 #100
        self.opacity_reset_interval = 3000 #3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000 #50_000
        self.densify_grad_threshold = 0.0002
        self.oneupSHdegree_step = 1000
        self.min_opacity = 0.01
        self.start_normal_reg = 8000
        self.lambda_dist = 1000

        self.deform_lr_scale = 1.
        self.gt_alpha_mask_as_scene_mask = False

        self.albedo_lr = 0.01
        self.albedo_after_stage1_lr = 0.01
        self.albedo_rest_lr = self.albedo_lr /20
        self.envmap_cubemap_lr = 0.1
        self.lambda_separation = 0.005 # for supervised  0.02 but tested long time ago
        self.d_color_reg_loss_weight = 0.01
        self.d_xyz_loss_weight = 0.001
        self.d_lower_hemisphere_weight = 0.00001 # mostly gives nicer looking envmap - no bright artifacts on lower hemisphere which is normally almost not supervised.
        self.lambda_alpha_loss = 0.1
        self.envmap_resolution = 32 #128
        self.envmap_init_value = 1.5
        self.envmap_activation = 'exp'

        self.train_ray = False
        self.trace_num_rays = (2**18)*1

        #its possible to test losses from irgs, but defaults to zero:
        self.lambda_roughness_smooth=0.0
        self.lambda_light=0.0
        self.lambda_light_smooth=0.0
        self.lambda_base_color_smooth=0.0

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    if not args_cmdline.model_path.endswith(args_cmdline.deform_type):
        args_cmdline.model_path = os.path.join(os.path.dirname(os.path.normpath(args_cmdline.model_path)), os.path.basename(os.path.normpath(args_cmdline.model_path)) + f'_{args_cmdline.deform_type}')

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

