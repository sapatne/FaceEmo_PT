import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T

class FERDataset(Dataset):
	def __init__(self, annotations_csv, img_dir, transform=None):
		img_labels = pd.read_csv(os.path.join('./data/', annotations_csv))
		self.img_labels = img_labels[img_labels['Folder'] == img_dir]
		self.img_dir = os.path.join('./data', img_dir)
		self.transform = transform

	def __len__(self):
		return len(self.img_labels)
	
	def __getitem__(self, index):
		img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 1])
		image = read_image(img_path)
		label = self.img_labels.iloc[index, 2]
		if self.transform:
			image = self.transform(image)

		return {'image': image, 'label': label}
	


train_dataset = FERDataset('annotations.csv', 'Training')
test_dataset = FERDataset('annotations.csv', 'Testing')
val_dataset = FERDataset('annotations.csv', 'Validation')


