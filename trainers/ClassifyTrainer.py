import os
from multiprocessing import Pool, Queue, Process

import scipy
import utils
import numpy as np

import torch
import torch.nn as nn
from .BaseTrainer import BaseTrainer
from utils import torch_downsample

from sklearn.metrics import f1_score, confusion_matrix, recall_score, jaccard_similarity_score, roc_curve, precision_recall_curve, roc_auc_score, auc

class ClassifyTrainer(BaseTrainer):
	def __init__(self, arg, G, torch_device, class_loss):
		super(ClassifyTrainer, self).__init__(arg, torch_device)
		self.class_loss = class_loss

		self.G = G
		self.lrG = arg.lrG
		self.beta = arg.beta
		self.fold = arg.fold
		self.optim = torch.optim.Adam(self.G.parameters(), lr = arg.lrG, betas = arg.beta)
		self.best_metric = 0
		self.sigmoid = nn.Sigmoid().to(self.torch_device)

		self.load()
		self.prev_epoch_loss = 0

	def save(self, epoch, filename = "models"):
		save_path = self.save_path +"/fold%s"%(self.fold)
		if os.path.exists(self.save_path) is False:
			os.mkdir(self.save_path)
		if os.path.exists(save_path) is False:
			os.mkdir(save_path)

		torch.save({"model_type": self.model_type,
			"start_epoch": epoch + 1,
			"network": self.G.state_dict(),
			"optimizer": self.optim.state_dict(),
			"best_metric": self.best_metric,
			}, save_path + "/%s.pth.tar" % (filename))
		print("Model saved %d epoch" % (epoch))

	def load(self):
		save_path = self.save_path + "/fold%s"%(self.fold)
		if os.path.exists(save_path + "/models.pth.tar") is True:
			print("Load %s File" % (save_path))
			ckpoint = torch.load(save_path + "/models.pth.tar")
			if ckpoint["model_type"] != self.model_type:
				raise ValueError("Ckpoint Model Type is %s" % (ckpoint["model_type"]))

			self.G.load_state_dict(ckpoint['network'])
			self.optim.load_state_dict(ckpoint['optimizer'])
			self.start_epoch = ckpoint['start_epoch']
			self.best_metric = ckpoint['best_metric']
			print("Load Model Type: %s, epoch: %d" % (ckpoint["model_type"], self.start_epoch))
		else:
			print("Load Failed, not exists file")

	def _init_model(self):
		for m in self.G.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()
		self.optim = torch.optim.Adam(self.G.parameters(), lr = self.lrG, betas = self.beta)

	def train(self, train_loader, val_loader = None):
		print("\nStart Train")

		for epoch in range(self.start_epoch, self.epoch):
			for i, (input_, target_, _) in enumerate(train_loader):
				self.G.train()
				input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
				output_ = self.G(input_)
				target_ = target_.long().squeeze(1)
				#print(input_.shape)
				#print(output_.shape)
				#print(target_.shape)
				class_loss = self.class_loss(output_, target_)

				self.optim.zero_grad()
				class_loss.backward()
				self.optim.step()

				if (i % 50) == 0:
					self.logger.will_write("[Train] epoch: %d loss: %f" % (epoch, class_loss))

			if val_loader is not None:
				self.valid(epoch, val_loader)
			else:
				self.save(epoch)
		print("End Train\n")

	def forward_for_test(self, input_, target_):
		input_ = input_.to(self.torch_device)
		output_ = self.G(input_)
		target_ = target_.to(self.torch_device)
		return input_, output_, target_

	def valid(self, epoch, val_loader):
		self.G.eval()
		with torch.no_grad():
			# (tn, fp, fn, tp)
			cm = utils.ConfusionMatrix()

			for i, (input_, target_, _) in enumerate(val_loader):
				input_ = input_.to(self.torch_device)
				output_ = self.G(input_)
				target_ = target_.to(self.torch_device)

				ground_truth = target_.int().squeeze(1)
				prediction = torch.argmax(output_, dim=1).int()

				cm.update(utils.confusion_matrix(prediction, ground_truth, reduce = False))
			metric = 1.5*cm.f2 + cm.accuracy
			if metric > self.best_metric:
				self.best_metric = metric
				self.save(epoch)


			self.logger.write("[Val] epoch: %d accuracy: %f f05: %f f1: %f f2: %f" % (epoch, cm.accuracy, cm.f05, cm.f1, cm.f2))

	def get_best_th(self, loader):
		y_true = np.array([])
		y_pred = np.array([])
		for i, (input_, target_, _) in enumerate(loader):
			input_, output_, target_ = self.forward_for_test(input_, target_)
			target_np = utils.slice_threshold(target_, 0.5)

			y_true = np.concatenate([y_true, target_np.flatten()], axis = 0)
			y_pred = np.concatenate([y_pred, output_.flatten()], axis = 0)


		pr_values = np.array(precision_recall_curve(y_true, y_pred))

		# To Do : F0.5 score
		f_best, th_best = -1, 0
		for precision, recall, threshold in zip(*pr_value):
			f05 = (5 * precision * recall) / (precision + (4 * recall))
			if f05 > f_best:
				f_best = f05
				th_best = threshold

		return f_best, th_best

	def test(self, test_loader, val_loader):
		print("\nStart Test")
		self.G.eval()
		with torch.no_grad():
			cm = utils.ConfusionMatrix()

			y_true = np.array([])
			y_pred = np.array([])

			for i, (input_, target_, f_name) in enumerate(test_loader):
				input_, output_, target_ = self.forward_for_test(input_, target_)
				ground_truth = target_.int().squeeze(1)
				prediction = torch.argmax(output_, dim=1).int()

				cm.update(utils.confusion_matrix(prediction, ground_truth, reduce=False))

				prediction_np = prediction.type(torch.FloatTensor).numpy()
				ground_truth_np = ground_truth.type(torch.FloatTensor).numpy()

				y_true = np.concatenate([y_true, ground_truth_np], axis = 0)
				y_pred = np.concatenate([y_pred, prediction_np], axis = 0)

				for batch_idx in range(0, input_.shape[0]):
					output_b = prediction_np[batch_idx]
					target_b = ground_truth_np[batch_idx]

					self.logger.will_write("[Test] fname:%s true_label:%f prediction:%f" % (f_name[batch_idx][:-4], target_b, output_b))

			pr_values = np.array(precision_recall_curve(y_true, y_pred))

			roc_auc = roc_auc_score(y_true, y_pred)
			pr_auc = auc(pr_values[0], pr_values[1], reorder=True)

		self.logger.write("accuracy:%f f05:%f f1:%f f2:%f roc:%f pr:%f" % (cm.accuracy, cm.f05, cm.f1, cm.f2, roc_auc, pr_auc))
		print("End Test\n")
