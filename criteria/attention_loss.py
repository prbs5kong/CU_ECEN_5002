# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from configs.paths_config import model_paths
from models.attentionmodule.MODELS.cbam import CBAM  # CBAM import


class AttentionLoss(nn.Module):

	def __init__(self):
		super(AttentionLoss, self).__init__()
		print("Loading CBAM model from path: {}".format(model_paths["cbam"]))
		self.model = self.__load_model()
		self.model.cuda()
		self.model.eval()
		#import ipdb
		#ipdb.set_trace()
		self.attention = CBAM(gate_channels=2048)  # initialize

	@staticmethod
	def __load_model():
		import torchvision.models as models
		model = models.__dict__["resnet50"]()
		#please solve the error RuntimeError: Given groups=1, weight of size [64, 3, 7, 7], expected input[4, 1, 224, 224] to have 3 channels, but got 1 channels instead
		#nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
		# freeze all layers but the last fc
		for name, param in model.named_parameters():
			if name not in ['fc.weight', 'fc.bias']:
				param.requires_grad = False
		model.fc = nn.Identity()
		checkpoint = torch.load(model_paths['cbam_pt'], map_location="cpu")
		state_dict = checkpoint['state_dict']
		# rename cbam pre-trained keys
		for k in list(state_dict.keys()):
			# retain only encoder_q up to before the embedding layer
			if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
				# remove prefix
				state_dict[k[len("module.encoder_q."):]] = state_dict[k]
			# delete renamed or unused k
			del state_dict[k]
		model.load_state_dict(state_dict, strict=False)
		#assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
		# remove output layer
		# Load missing parameters
		#import ipdb
		#ipdb.set_trace()
		model = nn.Sequential(*list(model.children())[:-1]).cuda()
		return model
    
	def extract_feats(self, x):
		x = F.interpolate(x, size=224)
		x_feats = self.model(x)
		x_feats = nn.functional.normalize(x_feats, dim=1)
		x_feats = x_feats.squeeze()
		return x_feats
	
    
	'''def forward(self, y_hat, y, x):
		y_attention = self.attention(y)
		x_attention = self.attention(x)
		y_hat_attention = self.attention(y_hat)
		if y_hat_attention.size() != y_attention.size():
			y_hat_attention = F.interpolate(y_hat_attention, size=y_attention.size()[-2:], mode="bilinear")
		loss_cbam = F.mse_loss(y_hat_attention, y_attention) + F.mse_loss(y_hat_attention, x_attention)
		# compute similarity improvement
		sim_improvement = (F.mse_loss(y_hat_feats, y_feats) - F.mse_loss(y_hat_attention, y_attention)).item()
		# identity logs
		id_logs = {"sim_improvement": sim_improvement}
		return loss_cbam, sim_improvement, id_logs'''

	def forward(self, y_hat, y, x):
		n_samples = x.shape[0]
		import torch
		import cv2
		import numpy as np
		import matplotlib.pyplot as plt

		# Assume 'tensor' is your [4,3,256,256] tensor
		#for i in range(x.shape[0]):
			#img = x[i].detach().cpu().numpy()    # convert it into numpy array
			#img = img.transpose((1, 2, 0))
			
			#if img.shape[2] == 1:
				#img = img.squeeze(axis=2)       # change [3,256,256] to [256,256,3]\
			#plt.imshow(img)
			#plt.savefig(f'a_{i}.png')

		# Assume 'tensor' is your [4,3,256,256] tensor
		#for i in range(y.shape[0]):
			#img = y[i].detach().cpu().numpy()    # convert it into numpy array
			#img = img.transpose((1, 2, 0))      # change [3,256,256] to [256,256,3]\
			#plt.imshow(img)
			#plt.savefig(f'b_{i}.png')

		# Assume 'tensor' is your [4,3,256,256] tensor
		#for i in range(y_hat.shape[0]):
			#img = y_hat[i].detach().cpu().numpy()    # convert it into numpy array
			#img = img.transpose((1, 2, 0))      # change [3,256,256] to [256,256,3]\
			#plt.imshow(img)
			#plt.savefig(f'c_{i}.png')

		#if x.shape[2] == 1:
		#	x = x.squeeze(axis=2)
		x = x.repeat_interleave(3, dim=1)
		x_feats = self.extract_feats(x) 
		y_feats = self.extract_feats(y)
		y_hat_feats = self.extract_feats(y_hat)

		y_hat_feats = F.interpolate(y_hat_feats.view(y_hat_feats.size(0), y_hat_feats.size(1), 1, 1), size=(7, 7), mode='bilinear', align_corners=False)
		y_hat_feats = F.interpolate(y_hat_feats, size=(7, 7), mode='bilinear', align_corners=False)
		attention_map = self.attention(y_hat_feats)

		y_feats = F.interpolate(y_feats.view(y_feats.size(0), y_feats.size(1), 1, 1), size=(7, 7), mode='bilinear', align_corners=False)
		y_feats = F.interpolate(y_feats, size=(7, 7), mode='bilinear', align_corners=False)
		
		x_attention = None
		#import ipdb
		#ipdb.set_trace()
		y_hat_attention = y_hat_feats * attention_map
		#if y_hat_attention.size() != y_attention.size():
			#y_hat_attention = y_hat_attention.view(y_hat_attention.size(0), y_hat_attention.size(1), 1, 1, 1, 1)
			#y_hat_attention = y_hat_attention.expand(-1, -1, y_attention.size(2), y_attention.size(3), -1, -1)
		#if y_hat_attention.size() != x_attention.size():
			#y_hat_attention = y_hat_attention.view(y_hat_attention.size(0), y_hat_attention.size(1), 1, 1, -1, -1)
			#y_hat_attention = y_hat_attention.expand(-1, -1, x_attention.size(2), x_attention.size(3), -1, -1)
		#y_attention = y_feats * attention_map.unsqueeze(-1).unsqueeze(-1)
		x_feats = F.interpolate(x_feats.view(x_feats.size(0), x_feats.size(1), 1, 1), size=(7, 7), mode='bilinear', align_corners=False)
		x_feats = F.interpolate(x_feats, size=(7, 7), mode='bilinear', align_corners=False)
		x_attention = x_feats * attention_map
		y_attention = y_feats * attention_map
		# CBAM loss
		#loss_cbam = F.cross_entropy(y_hat_logit, y) + F.cross_entropy(y_hat_logit, x)
		loss_cbam = F.mse_loss(y_hat_attention, y_attention) + F.mse_loss(y_hat_attention, x_attention)
		# compute similarity improvement
		sim_improvement = (F.mse_loss(y_hat_feats, y_feats) - F.mse_loss(y_hat_attention, y_attention)).item()
		# identity logs
		id_logs = {"sim_improvement": sim_improvement}
		#import ipdb
		#ipdb.set_trace()

		return loss_cbam, sim_improvement, id_logs
   
	def attention_map(self, y_hat, y, x):
		n_samples = x.shape[0]
		x_feats = self.extract_feats(x)
		y_feats = self.extract_feats(y)
		y_hat_feats = self.extract_feats(y_hat)

		# Make attention map using attention module
		attention_map = self.attention(y_hat_feats)

		y_hat_attention = y_hat_feats * attention_map.unsqueeze(-1).unsqueeze(-1)
		y_attention = y_feats * attention_map.unsqueeze(-1).unsqueeze(-1)
		x_attention = x_feats * attention_map

		# attention map + result
		return attention_map, y_hat_attention, y_attention, x_attention
