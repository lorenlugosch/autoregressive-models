import torch
import torch.utils.data
import torchaudio
import os
import soundfile as sf
import numpy as np
import configparser
import multiprocessing
import json
import pandas as pd
from subprocess import call

class Config:
	def __init__(self):
		pass

def read_config(config_file):
	config = Config()
	parser = configparser.ConfigParser()
	parser.read(config_file)

	#[experiment]
	config.seed=int(parser.get("experiment", "seed"))
	config.folder=parser.get("experiment", "folder")

	# Make a folder containing experiment information
	if not os.path.isdir(config.folder):
		os.mkdir(config.folder)
		os.mkdir(os.path.join(config.folder, "training"))
	call("cp " + config_file + " " + os.path.join(config.folder, "experiment.cfg"), shell=True)

	#[model]
	config.num_layers=int(parser.get("model", "num_layers"))
	config.num_hidden=int(parser.get("model", "num_hidden"))
	config.num_mel_bins=int(parser.get("model", "num_mel_bins"))
	config.frame_skipping=parser.get("model", "frame_skipping")=="True"

	#[training]
	config.base_path=parser.get("training", "base_path")
	config.lr=float(parser.get("training", "lr"))
	config.lr_period=int(parser.get("training", "lr_period"))
	config.gamma=float(parser.get("training", "gamma"))
	config.batch_size=int(parser.get("training", "batch_size"))
	config.num_epochs=int(parser.get("training", "num_epochs"))

	return config

def get_ASR_datasets(config):
	"""
	config: Config object (contains info about model and training)
	"""
	base_path = config.base_path

	# Get dfs
	train_df = pd.read_csv(os.path.join(base_path, "train_data.csv")) #"train-clean-360.csv")) #"train_data.csv"))
	valid_df = pd.read_csv(os.path.join(base_path, "valid_data.csv"))
	test_df = pd.read_csv(os.path.join(base_path, "test_data.csv"))

	# Create dataset objects
	train_dataset = ASRDataset(train_df, config)
	valid_dataset = ASRDataset(valid_df, config)
	test_dataset = ASRDataset(test_df, config)

	return train_dataset, valid_dataset, test_dataset

class ASRDataset(torch.utils.data.Dataset):
	def __init__(self, df, config):
		"""
		df: dataframe of wav file paths and transcripts
		config: Config object (contains info about model and training)
		"""
		# dataframe with wav file paths, transcripts
		self.df = df
		self.base_path = config.base_path
		self.loader = torch.utils.data.DataLoader(self, batch_size=config.batch_size, num_workers=multiprocessing.cpu_count(), shuffle=True, collate_fn=CollateWavsASR())

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		x, fs = sf.read(os.path.join(self.base_path, self.df.path[idx]))
		return (x, idx)

class CollateWavsASR:
	def __init__(self):
		self.max_length = 500000

	def __call__(self, batch):
		"""
		batch: list of tuples (input wav, output labels)

		Returns a minibatch of wavs and labels as Tensors.
		"""
		x = []; idxs = []
		batch_size = len(batch)
		for index in range(batch_size):
			x_,idx = batch[index]

			# throw away large audios
			if len(x_) < self.max_length:
				x.append(torch.tensor(x_).float())
				idxs.append(idx)

		batch_size = len(idxs) # in case we threw some away

		# pad all sequences to have same length
		T = [len(x_) for x_ in x]
		T_max = max(T)
		for index in range(batch_size):
			x_pad_length = (T_max - len(x[index]))
			x[index] = torch.nn.functional.pad(x[index], (0,x_pad_length))

		x = torch.stack(x)

		return (x,T,idxs)
