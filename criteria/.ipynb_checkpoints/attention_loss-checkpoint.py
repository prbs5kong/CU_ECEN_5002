# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from configs.paths_config import model_paths
from models.attentionmodule.MODELS.cbam import CBAM  # CBAM 모듈을 import합니다.


class AttentionLoss(nn.Module):

	def __init__(self):
		super(AttentionLoss, self).__init__()
		print("Loading CBAM model from path: {}".format(model_paths["cbam"]))
		self.model = self.__load_model()
		self.model.cuda()
		self.model.eval()
		#import ipdb
		#ipdb.set_trace()
		self.attention = CBAM(gate_channels=2048)  # CBAM 모듈을 초기화합니다.

	@staticmethod
	def __load_model():
		import torchvision.models as models
		model = models.__dict__["resnet50"]()
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
    
	''''def forward(self, y_hat, y, x):
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
		x_feats = self.extract_feats(x)
		y_feats = self.extract_feats(y)
		y_hat_feats = self.extract_feats(y_hat)
		y_hat_feats = y_hat_feats.view(y_hat_feats.size(0), y_hat_feats.size(1), 1, 1)
		attention_map = self.attention(y_hat_feats)
		y_hat_attention = y_hat_feats * attention_map.unsqueeze(-1).unsqueeze(-1)
		if y_hat_attention.size() != y_attention.size():
			y_hat_attention = y_hat_attention.view(y_hat_attention.size(0), y_hat_attention.size(1), 1, 1, 1, 1)
			y_hat_attention = y_hat_attention.expand(-1, -1, y_attention.size(2), y_attention.size(3), -1, -1)
		if y_hat_attention.size() != x_attention.size():
			y_hat_attention = y_hat_attention.view(y_hat_attention.size(0), y_hat_attention.size(1), 1, 1, -1, -1)
			y_hat_attention = y_hat_attention.expand(-1, -1, x_attention.size(2), x_attention.size(3), -1, -1)
		y_attention = y_feats * attention_map.unsqueeze(-1).unsqueeze(-1)
		x_attention = x_feats * attention_map
		# CBAM loss 계산
		loss_cbam = F.mse_loss(y_hat_attention, y_attention) + F.mse_loss(y_hat_attention, x_attention)
		# compute similarity improvement
		sim_improvement = (F.mse_loss(y_hat_feats, y_feats) - F.mse_loss(y_hat_attention, y_attention)).item()
		# identity logs
		id_logs = {"sim_improvement": sim_improvement}
		import ipdb
		ipdb.set_trace()
		return loss_cbam, sim_improvement, id_logs
   
	def attention_map(self, y_hat, y, x):
		n_samples = x.shape[0]
		x_feats = self.extract_feats(x)
		y_feats = self.extract_feats(y)
		y_hat_feats = self.extract_feats(y_hat)

		# CBAM 모듈을 이용하여 attention map을 생성합니다.
		attention_map = self.attention(y_hat_feats)

		# attention map과 원본 이미지를 곱하여 attention이 적용된 이미지를 생성합니다.
		y_hat_attention = y_hat_feats * attention_map.unsqueeze(-1).unsqueeze(-1)
		y_attention = y_feats * attention_map.unsqueeze(-1).unsqueeze(-1)
		x_attention = x_feats * attention_map

		# attention map과 각 이미지에 attention을 적용시킨 결과를 반환합니다.
		return attention_map, y_hat_attention, y_attention, x_attention
