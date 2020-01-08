import torch
import numpy as np
from models import AutoregressiveModel
from data import get_ASR_datasets, read_config
from training import Trainer
import argparse

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='run training')
parser.add_argument('--restart', action='store_true', help='load checkpoint from a previous run')
parser.add_argument('--config_path', type=str, help='path to config file with hyperparameters, etc.')
args = parser.parse_args()
train = args.train
restart = args.restart
config_path = args.config_path

# Read config file
config = read_config(config_path)
torch.manual_seed(config.seed); np.random.seed(config.seed)

# Initialize model
model = AutoregressiveModel(config=config)
print(model)

# Generate datasets
train_dataset, valid_dataset, test_dataset = get_ASR_datasets(config)

trainer = Trainer(model=model, config=config)
if restart: trainer.load_checkpoint()

#######################
if not train:
	from data import CollateWavsASR
	import matplotlib.pyplot as plt
	c = CollateWavsASR()
	indices = [0,1]
	b = [ train_dataset.__getitem__(idx) for idx in indices]
	batch = c.__call__(b)
	x,T,idxs = batch
	encoded, predicted, diff = model(x,T)
	fbank = model.compute_fbank((x,T))
	plt.subplot(2,1,1); plt.imshow(predicted[0].detach().transpose(0,1)); plt.subplot(2,1,2); plt.imshow(fbank[0].detach().transpose(0,1)); plt.show()
	t = 3; ((diff[0]**2).mean(1).detach() > t).sum(); plt.plot((diff[0]**2).mean(1).detach() > t); plt.show()
#######################


# Train the final model
if train:
	for epoch in range(config.num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.num_epochs))
		train_loss = trainer.train(train_dataset)
		valid_loss = trainer.test(valid_dataset, set="valid")

		print("========= Results: epoch %d of %d =========" % (epoch+1, config.num_epochs))
		print("train loss: %.2f| valid loss: %.2f\n" % (train_loss, valid_loss) )

		trainer.save_checkpoint()

	trainer.load_best_model()
	test_loss = trainer.test(test_dataset, set="test")
	print("========= Test results =========")
	print("test loss: %.2f \n" % (test_loss) )
