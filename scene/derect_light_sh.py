import torch
import torch.nn as nn
from arguments import OptimizationParams


class DirectLightEnv:

    def __init__(self, sh_degree):
        self.default_shs_dc = None
        self.default_shs_rest = None
        self.sh_degree = sh_degree
        env_shs = torch.zeros((1, 3, (self.sh_degree + 1) ** 2)).float().cuda()
        self.env_shs_dc = nn.Parameter(env_shs[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self.env_shs_rest = nn.Parameter(env_shs[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

    @property
    def get_env_shs(self):
        shs_dc = self.env_shs_dc
        shs_rest = self.env_shs_rest
        return torch.cat((shs_dc, shs_rest), dim=1)
    
    def modify_shs(self, index):
        if(self.default_shs_dc is None):
            self.default_shs_dc = self.env_shs_dc.clone()
            self.default_shs_rest = self.env_shs_rest.clone()
        
        if(index == 0):
            self.env_shs_dc = self.default_shs_dc.clone()
            self.env_shs_rest = self.default_shs_rest.clone()
            return
        
        self.env_shs_dc = self.env_shs_dc.cpu()
        self.env_shs_rest = self.env_shs_rest.cpu()
        self.env_shs_dc[0] = self.env_shs_rest[0][index]
        self.env_shs_dc = self.env_shs_dc.cuda()
        self.env_shs_rest = self.env_shs_rest.cuda()

    @property
    def get_direct_color(self):
        return [self.env_shs_dc[0][0][0].item(), self.env_shs_dc[0][0][1].item(), self.env_shs_dc[0][0][2].item()]
    
    def set_direct_color(self, rgb):
        print(self.env_shs_rest)
        print(self.env_shs_rest.shape)
        self.env_shs_dc = self.env_shs_dc.cpu()
        self.env_shs_dc[0][0][0] = rgb[0]
        self.env_shs_dc[0][0][1] = rgb[1]
        self.env_shs_dc[0][0][2] = rgb[2]
        self.env_shs_dc = self.env_shs_dc.cuda()

    def training_setup(self, training_args: OptimizationParams):
        if training_args.env_rest_lr < 0:
            training_args.env_rest_lr = training_args.env_lr / 20.0
        l = [
            {'params': [self.env_shs_dc], 'lr': training_args.env_lr, "name": "env_dc"},
            {'params': [self.env_shs_rest], 'lr': training_args.env_rest_lr, "name": "env_rest"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def capture(self):
        captured_list = [
            self.sh_degree,
            self.env_shs_dc,
            self.env_shs_rest,
            self.optimizer.state_dict(),
        ]

        return captured_list

    def restore(self, model_args, training_args,
                is_training=False, restore_optimizer=True):
        pass

    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path)
        (self.sh_degree,
         self.env_shs_dc,
         self.env_shs_rest,
         opt_dict) = model_args[:4]

        if restore_optimizer:
            try:
                self.optimizer.load_state_dict(opt_dict)
            except:
                print("Not loading optimizer state_dict!")

        return first_iter
