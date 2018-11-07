import utils
import os
from logger import logger

class BaseTrainer:
    def __init__(self, arg, torch_device):
        self.torch_device = torch_device 
        
        self.model_type = arg.model

        self.z_idx = 0

        self.epoch = arg.epoch
        self.start_epoch = 0

        self.batch_size = arg.batch_size
        
        self.save_path = utils.get_save_dir(arg)
        self.log_file_path = self.save_path+"/fold%s/log.txt"%(arg.fold)

        if os.path.exists(self.save_path) is False:
            os.mkdir(self.save_path)

        if os.path.exists(self.save_path + "/fold%s"%(arg.fold)) is False:
            os.mkdir(self.save_path + "/fold%s"%(arg.fold))

        #self.logger = logger.info(arg, self.save_path + "/fold%s"%(arg.fold))
    
    def save(self):
        raise NotImplementedError("notimplemented save method")

    def load(self):
        raise NotImplementedError("notimplemented save method")

    def train(self):
        raise NotImplementedError("notimplemented save method")

    def valid(self):
        raise NotImplementedError("notimplemented valid method")

    def test(self):
        raise NotImplementedError("notimplemented test method")

    def inference(self):
        raise NotImplementedError("notimplemented interence method")
