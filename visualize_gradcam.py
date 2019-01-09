import os
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
from loader import loader
import argparse
from models.dense_net import DenseNet
from models.latefuse_dense_net import LateFuse_DenseNet
from models.GradCam import GradCam
from models.LF_GradCam import LF_GradCam
import numpy as np

if __name__ == "__main__":
	os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
	save_path = '/home/intern/osteo_classification/outs/fold_DenseNet_acc_m2'
	ckpoint = torch.load(save_path + "/models.pth.tar")
	torch_device = torch.device("cuda")
	log_path = save_path + '/TP.txt'
	f = open(log_path, 'r')
	lines = f.readlines()
	lines = np.array(lines)
	lines = np.char.rstrip(lines)

	model = DenseNet(growthRate = 12, depth = 40, reduction = 0.5, bottleneck = True, nClasses = 2)
	model = nn.DataParallel(model).to(torch_device)
	model.load_state_dict(ckpoint['network'])
	grad_cam = GradCam(model=model.module, hooks = ["dense2"], device=torch_device)

	test_path= "/home/intern/osteo_classification/data/test/"
	test_loader = loader(test_path, 1, transform = None, sampler= '', torch_type = 'float', cpus = 32, shuffle = False, drop_last = True, late_fusion = False)
	for i, (input_, target_, f_name) in enumerate(test_loader):
		file_name = f_name[0].split('.npy')[0]
		if np.any(lines == file_name) :
			input_.requires_grad_(True)
			target_index = None
			cam_mask, _ = grad_cam(input_, target_index)
    
			img = input_[0].permute(1,2,0).detach().numpy()
			cam_img = LF_GradCam.cam_on_image(img, cam_mask)
			file_name = save_path + '/CAM_mask_' + f_name[0].split('.npy')[0] + '.npy'
			np.save(file_name, cam_img)
