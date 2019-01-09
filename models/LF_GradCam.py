# Author : Daewoong Ahn
import copy

import torch
import torch.nn.functional as F
from torch.autograd import Function

import cv2
import numpy as np
from skimage.transform import resize as skresize
from torchvision.transforms import Resize as TorchVisionResize



class LF_GradCam:
    def __init__(self, model, hooks, device):
        if isinstance(hooks, list) is False:
            raise ValueError("Hooks must be list of str")
        
        self.device = device
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.model = self.model.to(device)

        self.feature = []
        self.grad = []

        # Setting Hook
        hook_layer = self.model
        for h in hooks:
            if isinstance(h, str) is False:
                raise ValueError("Hooks must be list of str")
            if h not in hook_layer._modules.keys():
                raise ValueError("Hook[{}] is not in {}".format(h, hook_layer._modules.keys()))
            hook_layer = hook_layer._modules[h]

        def save_feature(module, input_, output):
            self.feature.append(output.detach())

        def save_grad(module, grad_in, grad_out):
            self.grad.append(grad_out[0].detach())

        self.grad_hook = hook_layer.register_backward_hook(save_grad)
        self.feature_hook = hook_layer.register_forward_hook(save_feature)

    @staticmethod
    def get_layer_name(model):
        for n, m in model._modules.items():
            print("Name : ", n, "\nModule : ", m)

    def remove_hook_data(self):
        self.feature = []
        self.grad = []

    def backward(self, output):
        _, idx = output.max(dim=0)
        one_hot = torch.zeros_like(output).to(self.device)
        one_hot[idx] = 1
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
    
    
    def gunho(self, grad, feature):
        print(grad.shape)
        alpha = torch.sum(grad, dim=(1, 2, 3), keepdim=True)
        gcam = (alpha * grad).sum(dim=0)
        gcam = gcam.detach().cpu().numpy()
        print('gcam shape', gcam.shape)
        return gcam

    def gdcam(self, grad, feature):
        grad = torch.unsqueeze(grad, 0)
        weights = F.adaptive_avg_pool3d(grad, 1)
        weights.squeeze_(0)

        gcam = (feature * weights).sum(dim=0)
        gcam = torch.clamp(gcam, min=0.)
        gcam = gcam.detach().cpu().numpy()
        return gcam
    
    def gdcampp(self, grad, feature):
        grad_2 = grad * grad
        grad_3 = grad * grad_2
        weights = grad_2 / ((2 * grad_2) + (grad_3 * feature) + 1e-12)
        
        weights.unsqueeze_(0)
        weights = F.adaptive_avg_pool3d(weights, 1)
        weights.squeeze_(0)
        
        feature = torch.clamp(feature, min=0.)
        gcam = (feature * weights).sum(dim=0)
        gcam = torch.clamp(gcam, min=0.)
        gcam = gcam.detach().cpu().numpy()
        return gcam
    
    def respondcam(self, grad, feature):
        weights = (grad * feature).sum(dim=(1,2,3)) / (feature.sum(dim=(1,2,3)) + 1e-8)
        weights.unsqueeze_(1).unsqueeze_(2).unsqueeze_(3)        
        gcam = (feature * weights).sum(dim=0)
        gcam = torch.abs(gcam)
        gcam = torch.sigmoid(gcam)
        gcam = gcam.detach().cpu().numpy()
        return gcam
    

    def __call__(self, input_, clinic_, index=None, mode="gunho"):
        output = self.model(input_.to(self.device), clinic_.to(self.device))
        _, pred_idx = output.max(dim=0)
        self.backward(output)

        # idx : [-1] : last output, [0] : remove batch dim
        feature_map = self.feature[-1]
        grad = self.grad[-1]
        if mode == "gunho":
            gcam = self.gunho(grad, feature_map)
        elif mode == "gdcam":
            gcam = self.gdcam(grad, feature_map)
        elif mode == "gdcampp":
            gcam = self.gdcampp(grad, feature_map)
        elif mode == "respond":
            gcam = self.respondcam(grad, feature_map)
        
        gcam -= gcam.min() ; gcam /= gcam.max()
        resize_shape = np.array([1, input_.shape[-2], input_.shape[-1]])
        gcam = skresize(gcam, resize_shape, mode="constant")
        gcam = gcam - np.min(gcam)
        gcam = gcam / np.max(gcam)
        gcam = np.float32(gcam)
        gcam = np.squeeze(gcam)
        return gcam, pred_idx
    
    def multicam(self, input_, index=None):
        output = self.model(input_.to(self.device))[0]
        _, pred_idx = output.max(dim=1)
        self.backward(output)

        # idx : [-1] : last output, [0] : remove batch dim
        feature_map = self.feature[-1][0]
        grad = self.grad[-1][0]
        gunho = self.gunho(grad, feature_map)
        gd = self.gdcam(grad, feature_map)
        gdpp = self.gdcampp(grad, feature_map)
        resp = self.respondcam(grad, feature_map)
        def _resize(gcam):
            gcam -= gcam.min() ; gcam /= gcam.max()
            gcam = skresize(gcam, input_.shape[-3:], mode="constant")
            gcam -= gcam.min() ; gcam /= gcam.max()
            return gcam
        resp = _resize(resp)
        gd = _resize(gd)
        gdpp = _resize(gdpp)
        gunho = _resize(gunho)
        return resp, gd, gdpp, gunho, pred_idx
    @staticmethod
    def cam_on_image(img, mask):
        if img.shape[-1] > 1:
            img = img[:, :, 0]
        img = img.reshape(704, 457, 1)
        img = np.concatenate([img, img, img], axis=2)
        print(img.shape)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)
