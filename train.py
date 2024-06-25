from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import torch
from torchsummary import summary

# from load_dataset import train_loader, val_loader
from models import VGG16

def main():
	emotion_table = {	
		'neutral'  : 0, 
		'happiness': 1, 
		'surprise' : 2, 
		'sadness'  : 3, 
		'anger'    : 4, 
		'disgust'  : 5, 
		'fear'     : 6, 
		'contempt' : 7
	}

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	num_epochs = 50
	num_classes = 7
	model = VGG16(num_classes).to(device)
	optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.005)

	summary(model, (1, 48, 48))

if __name__ == "__main__":
	main()