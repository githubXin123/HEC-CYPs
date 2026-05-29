import os
import json
from torch import device
from torch.cuda import is_available

class Configurator:
    def __init__(self, config_path=None):
        self.config_path = config_path
    
    def configurate(self):
        if self.config_path is not None:
            with open(self.config_path, 'r') as f:
                config_dct = json.load(f)
            self.__dict__.update(config_dct)

        self.prj_dir = os.path.join('projects', self.prj_name)

        if os.path.exists(self.prj_dir):
            self.log_dir = os.path.join(self.prj_dir, 'logs')
            self.data_dir = os.path.join(self.prj_dir, 'datasets')
            return
        else:
            self.log_dir = os.path.join(self.prj_dir, 'logs')
            self.data_dir = os.path.join(self.prj_dir, 'datasets')
            os.makedirs(self.prj_dir)
            os.makedirs(self.log_dir)
            os.makedirs(self.data_dir)
    
    @staticmethod
    def check_device():
        if is_available():
            DEVICE = device("cuda")
        else:
            DEVICE = device("cpu")
        
        return DEVICE
