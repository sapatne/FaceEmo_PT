import torch
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_confusion_matrix, multiclass_accuracy
from models_fer import VGG13
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from customdataset import test_dataset
from utils import load_model

def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Device: {device}")

	emotions = ()

	test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
	model = VGG13(7)

	last_chk_pt = load_model('best_chkpt_5')
	if (last_chk_pt is not None):
		last_chk_pt = torch.load(last_chk_pt)
		model.load_state_dict(last_chk_pt['model_state_dict'])
	model.to(device)
	model.eval()

	outputs = []
	labels = []
	test_acc = []
	with torch.no_grad():
		for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
			image = sample['image'].to(device)
			label = sample['label'].to(device)
			labels.extend(label)
			output = model(image)
			outputs.extend(output)
			acc = multiclass_accuracy(output, label, num_classes=7)
			test_acc.append(acc)
	print("Avg Test Accuracy: ", torch.mean(torch.stack(test_acc, 0)))
	confusion_matrix(labels, outputs)
	# print(labels)
	# print()
if __name__ == "__main__":
	main()


