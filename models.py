import torch
import torchaudio
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

class AutoregressiveModel(torch.nn.Module):
	def __init__(self, config):
		super(AutoregressiveModel, self).__init__()
		self.compute_fbank = ComputeFBANK(config)
		self.encoder = Encoder(config)
		self.regressor = torch.nn.Linear(self.encoder.out_dim, config.num_mel_bins)

	def load_pretrained(self, model_path=None):
		if model_path == None:
			model_path = os.path.join(self.checkpoint_path, "model_state.pth")
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.load_state_dict(torch.load(model_path, map_location=device))

	def forward(self, x, T):
		# move inputs to GPU
		if next(self.parameters()).is_cuda:
			x = x.cuda()

		fbank = self.compute_fbank((x,T))
		encoder_out = self.encoder.forward(fbank, T)
		x_predicted = self.regressor.forward(encoder_out)
		diff = x_predicted[:,:-1,:] - fbank

		return encoder_out, x_predicted, diff

class Encoder(torch.nn.Module):
	def __init__(self, config):
		super(Encoder, self).__init__()
		self.layers = []

		# input is FBANK
		layer = FBANKNormalizer(config)
		self.layers.append(layer)
		out_dim = config.num_mel_bins

		# pad to be causal
		layer = CausalPad()
		self.layers.append(layer)

		for idx in range(config.num_layers):
			# recurrent
			layer = torch.nn.GRU(input_size=out_dim, hidden_size=config.num_hidden, batch_first=True, bidirectional=False)
			self.layers.append(layer)
			out_dim = config.num_hidden

			# grab hidden states for each timestep
			layer = RNNOutputSelect()
			self.layers.append(layer)

			# dropout
			layer = torch.nn.Dropout(p=0.5)
			self.layers.append(layer)

			# fully-connected
			layer = torch.nn.Linear(out_dim, config.num_hidden)
			out_dim = config.num_hidden
			self.layers.append(layer)
			layer = torch.nn.LeakyReLU(0.125)
			self.layers.append(layer)

		self.layers = torch.nn.ModuleList(self.layers)
		self.out_dim = out_dim

	def forward(self, x, T):
		out = x
		for layer in self.layers:
			#print(out.shape, out.min(), out.max())
			out = layer(out)

		return out


class RNNOutputSelect(torch.nn.Module):
	def __init__(self):
		super(RNNOutputSelect, self).__init__()

	def forward(self, input):
		return input[0]

class NCL2NLC(torch.nn.Module):
	def __init__(self):
		super(NCL2NLC, self).__init__()

	def forward(self, input):
		"""
		input : Tensor of shape (batch size, T, Cin)
		Outputs a Tensor of shape (batch size, Cin, T).
		"""

		return input.transpose(1,2)

class CausalPad(torch.nn.Module):
	def __init__(self):
		super(CausalPad, self).__init__()

	def forward(self, input):
		return torch.nn.functional.pad(input, (0,0,1,0))

class FBANKNormalizer(torch.nn.Module):
	def __init__(self, config):
		super(FBANKNormalizer,self).__init__()
		self.num_mel_bins = config.num_mel_bins
		self.weight = torch.nn.Parameter(torch.tensor([1/10] * self.num_mel_bins))
		self.bias = torch.nn.Parameter(torch.tensor([0.] * self.num_mel_bins))

	def forward(self, fbank):
		out = fbank + self.bias.unsqueeze(0)
		out = out * self.weight.unsqueeze(0)
		return out

class ComputeFBANK(torch.nn.Module):
	def __init__(self, config):
		super(ComputeFBANK,self).__init__()
		self.num_mel_bins = config.num_mel_bins
		self.fbank_params = {
                    "channel": 0,
                    "dither": 0.0,
                    "window_type": "hanning",
                    "num_mel_bins":self.num_mel_bins,
                    "remove_dc_offset": False,
                    "round_to_power_of_two": False,
                    "sample_frequency":16000.0,
                }
		self.frame_skipping=config.frame_skipping

	def forward(self, input):
		"""
		input : (x,T)
		x : waveforms
		T : durations
		returns : (normalized) FBANK feature vectors
		"""
		fbanks = []
		x,T = input
		batch_size = len(x)
		for idx in range(batch_size):
			fbank_ = torchaudio.compliance.kaldi.fbank(x[idx].unsqueeze(0), **self.fbank_params)
			fbanks.append(fbank_)

		fbank = torch.stack(fbanks)
		if self.frame_skipping:
			return fbank[:,::2,:]
		else:
			return fbank

