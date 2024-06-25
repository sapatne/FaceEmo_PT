from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from customdataset import train_dataset, val_dataset
from models import VGG16

def save_model(epoch, model, optimizer):
	chkpoint_path = f'./checkpoints/chkpt_{epoch}'
	os.makedirs('/checkpoints',exist_ok=True)
	torch.save({
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
	}, chkpoint_path)

def load_model():
	path = './checkpoints/'
	if os.path.exists(path):
		all_chkpts = os.listdir(path)
		all_paths = [os.path.join(path, basename) for basename in all_chkpts]
		latest_chkpt = max(all_paths, key=os.path.getctime)
		return latest_chkpt
	
	return None


def main():
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)
	epoch = -1
	num_epochs = 20
	num_classes = 7

	train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

	model = VGG16(num_classes)
	optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.005)
	loss_func = CrossEntropyLoss()

	# load model
	print("Searching for saved model...")
	last_chkpt_path = load_model()
	if (last_chkpt_path is not None):
		last_chkpt = torch.load(last_chkpt_path)
		epoch = last_chkpt['epoch']
		print(f"Found Checkpoint! Resuming from {epoch}")
		model.load_state_dict(last_chkpt['model_state_dict'])
		optimizer.load_state_dict(last_chkpt['optimizer_state_dict'])
	else:
		print('Checkpoint not found. Starting from 0')

	model.to(device)
	# summary(model, (1, 256, 256))

	total_steps = len(train_loader)
	print(f'Total Steps per Epoch: {total_steps}')

	# SystemExit(0)
	for epoch in range(epoch + 1, num_epochs):
		print(f'Epoch: {epoch}')
		for i, data in tqdm(enumerate(train_loader)):
			image = data['image'].to(device)
			label = data['label'].to(device)

			output = model(image)
			loss = loss_func(output, label)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_steps, loss.item()))
		
		save_model(epoch, model, optimizer)

		# Validation
		with torch.no_grad():
			correct = 0
			total = 0
			for i, data in tqdm(enumerate(val_loader)):
				images = data['image'].to(device)
				labels = data['label'].to(device)
				outputs = model(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
				del images, labels, outputs
		
			print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total)) 
		


if __name__ == "__main__":
	main()