from torch.nn import CrossEntropyLoss, NLLLoss, utils
from torch.optim import AdamW, SGD, Adam
import torch
from torchsummary import summary
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler 
from tqdm import tqdm
import os
from torch.utils.tensorboard.writer import SummaryWriter
from torcheval.metrics.functional import multiclass_accuracy


from customdataset import train_dataset, val_dataset
from models_fer import VGG16, VGG13, VGGModel2
from utils import save_model, load_model, get_linear_schedule_with_warmup

def main():
	writer = SummaryWriter('runs/VGG16_FER2013 experiments_8')
	print("Training VGG16 Model on FER2013 Dataset..")

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Device: {device}')
	epoch = -1
	patience = 10
	num_epochs = 50
	num_classes = 7

	train_loader = DataLoader(train_dataset, batch_size=64, sampler=RandomSampler(train_dataset))
	val_loader = DataLoader(val_dataset, batch_size=64, sampler=SequentialSampler(val_dataset))

	model = VGG13(num_classes)
	model.to(device)
	optimizer = AdamW(model.parameters(), lr=4e-5, eps=1e-8)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=len(train_loader)*num_epochs)
	loss_func = CrossEntropyLoss()
	
	# load model
	print("Searching for saved model...")
	last_chkpt_path = load_model('best_chkpt_8')
	if (os.path.exists(last_chkpt_path)):
		last_chkpt = torch.load(last_chkpt_path)
		epoch = last_chkpt['epoch']
		print(f"Found Checkpoint! Previous Loss:: {last_chkpt['prev_best_loss']} Resuming from {epoch}")
		model.load_state_dict(last_chkpt['model_state_dict'])
		optimizer.load_state_dict(last_chkpt['optimizer_state_dict'])
		scheduler.load_state_dict(last_chkpt['scheduler_state_dict'])
	else:
		save_model('best_chkpt_8', epoch, model, optimizer, scheduler)
		print('Checkpoint not found. Starting from 0')

	writer.add_graph(model, train_dataset[3]['image'].unsqueeze(0).to(device))
	writer.flush()
	summary(model, (1, 48, 48), 64)


	total_steps = len(train_loader)
	print(f'Total Steps per Epoch: {total_steps}')
	
	best_val_acc = 0
	best_val_loss = 1e2
	train_losses = []
	val_accuracies = []
	val_losses = []
	train_accuracies = []
	# SystemExit(0)
	total_epochs = 0
	epoch = 0
	# while epoch < patience:
	for epoch in range(num_epochs):
		# print(f'Training model until {patience} epochs with no improvement.')
		print(f'Epoch: {epoch+1}, Total: {num_epochs}')
		model.train(True)
		train_loss = 0
		train_acc = 0
		for i, data in tqdm(enumerate(train_loader), ncols=100, total=total_steps):
			optimizer.zero_grad(True)
			image = data['image'].to(device)
			label = data['label'].to(device)

			output = model(image)
			train_acc += multiclass_accuracy(output, label, num_classes=num_classes)
			# _, predicted = torch.max(output.data, 1)
			loss = loss_func(output, label)
			loss.backward()
			utils.clip_grad_norm_(model.parameters(), 1.0)
			train_loss += loss

			optimizer.step()
			scheduler.step()

		avg_train_loss = train_loss / len(train_loader)
		train_losses.append(avg_train_loss)

		avg_train_acc = train_acc / len(train_loader)
		train_accuracies.append(avg_train_acc)
		print ('Epoch [{} ({})], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_steps, avg_train_loss, avg_train_acc))
		
		model.eval()
		# model.train(False)
		# Validation
		val_loss = 0
		val_acc = 0
		with torch.no_grad():
			for i, data in tqdm(enumerate(val_loader), ncols=100, total=len(val_loader)):
				images = data['image'].to(device)
				labels = data['label'].to(device)
				outputs = model(images)
				val_loss += loss_func(outputs, labels)
				val_acc += multiclass_accuracy(outputs, labels, num_classes=num_classes)
				del images, labels, outputs

			avg_val_loss = val_loss / len(val_loader)
			val_losses.append(avg_val_loss)

			avg_val_acc = val_acc / len(val_loader)
			val_accuracies.append(avg_val_acc)
			if avg_val_acc >= best_val_acc:
				print(f"Validation accuracy improved! Prev: {best_val_acc}, Improved: {avg_val_acc}")
				if avg_val_loss <= best_val_loss:
					print(f"Validation loss improved! Prev: {best_val_loss}, Improved: {avg_val_loss}")
					best_val_acc = avg_val_acc
					best_val_loss = avg_val_loss
					save_model('best_chkpt_8', total_epochs, model, optimizer, scheduler, best_val_loss, train_losses, val_losses, val_accuracies, train_accuracies)
					# print('Resetting epochs')
					# epoch = -1
				else:
					print("Prev loss lower than current loss. Not updating checkpoint..")
			
			print('Accuracy of the network on the {} validation images: {:.4f}.\nEpoch: {}, Best Validation Accuracy: {:.4f}\n Validation Loss: {:.4f}'.format(len(val_loader), avg_val_acc, epoch, best_val_acc, avg_val_loss)) 
		# epoch += 1
		# total_epochs += 1 
		writer.add_scalars('Plot losses', {'training': avg_train_loss, 'validation': avg_val_loss}, epoch)
		writer.flush()
		writer.add_scalars('Plot Accuracy', {'training': avg_train_acc, 'validation': avg_val_acc}, epoch)
		writer.flush()
	writer.close()


if __name__ == "__main__":
	main()