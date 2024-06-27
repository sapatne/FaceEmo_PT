from torch.nn import Module, Conv2d, Sequential, BatchNorm1d, BatchNorm2d, ReLU, MaxPool2d, Dropout, Linear, Softmax, LogSoftmax, Flatten

class VGG13(Module):
	def __init__(self, num_classes=7):
		super(VGG13, self).__init__()
		self.cnn1_1 = Sequential(
			Conv2d(1, 48, kernel_size=3),
			ReLU(),
			BatchNorm2d(48)
		)
		self.cnn1_2 = Sequential(
			Conv2d(48, 96, kernel_size=3),
			ReLU(),
			BatchNorm2d(96),
			MaxPool2d(kernel_size=2, stride=2),
			Dropout(0.25)
		)
		self.cnn2_1 = Sequential(
			Conv2d(96, 192, kernel_size=3),
			ReLU(),
			BatchNorm2d(192)
		)
		self.cnn2_2 = Sequential(
			Conv2d(192, 192, kernel_size=3),
			ReLU(),
			BatchNorm2d(192),
			MaxPool2d(kernel_size=2, stride=2),
			Dropout(0.25)
		)
		self.cnn3_1 = Sequential(
			Conv2d(192, 256, kernel_size=3),
			ReLU(),
			BatchNorm2d(256)
		)
		self.cnn3_2 = Sequential(
			Conv2d(256, 64, kernel_size=3),
			ReLU(),
			BatchNorm2d(64),
			MaxPool2d(kernel_size=2, stride=2),
			Dropout(0.25)
		)
		self.fcn = Sequential(
			Linear(256, 64),
			ReLU(),
			BatchNorm1d(64),
			Dropout(0.5)
		)
		self.out = Sequential(
			Linear(64, num_classes),
			# Softmax(dim=1)
		)

	def forward(self, x):
		out = self.cnn1_1(x)
		out = self.cnn1_2(out)
		out = self.cnn2_1(out)
		out = self.cnn2_2(out)
		out = self.cnn3_1(out)
		out = self.cnn3_2(out)
		out = out.reshape(out.size(0), -1)
		out = self.fcn(out)
		out = self.out(out)
		return out

class VGG16(Module):
	def __init__(self, num_classes=7):
		super(VGG16, self).__init__()
		self.cnn1_1 = Sequential(
			Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(32),
			ReLU()
		)
		self.cnn1_2 = Sequential(
			Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(64),
			ReLU(),
			MaxPool2d(kernel_size=2, stride=2)
		)
		self.cnn2_1 = Sequential(
			Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(128),
			ReLU()
		)
		self.cnn2_2 = Sequential(
			Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(128),
			ReLU(),
			MaxPool2d(kernel_size=2, stride=2)
		)
		self.cnn3_1 = Sequential(
			Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(256),
			ReLU(),
		)
		self.cnn3_2 = Sequential(
			Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(256),
			ReLU(),
		)
		self.cnn3_3 = Sequential(
			Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(256),
			ReLU(),
			MaxPool2d(kernel_size=2, stride=2)
		)
		self.cnn4_1 = Sequential(
			Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(512),
			ReLU(),
		)
		self.cnn4_2 = Sequential(
			Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(512),
			ReLU(),
		)
		self.cnn4_3 = Sequential(
			Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(512),
			ReLU(),
			MaxPool2d(kernel_size=2, stride=2)
		)
		self.cnn5_1 = Sequential(
			Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(512),
			ReLU(),
		)
		self.cnn5_2 = Sequential(
			Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(512),
			ReLU(),
		)
		self.cnn5_3 = Sequential(
			Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(512),
			ReLU(),
			MaxPool2d(kernel_size=2, stride=2)
		)
		self.fcn1 = Sequential(
			# Dropout(0.25),
			Linear(512, 512),
			ReLU()
		)
		self.fcn2 = Sequential(
			Dropout(0.25),
			Linear(512, 1024),
			ReLU()
		)
		self.out = Sequential(
			Dropout(0.5),
			Linear(1024, num_classes),
			Softmax(dim=1)
		)

	def forward(self, x):
		out = self.cnn1_1(x)
		out = self.cnn1_2(out)
		out = self.cnn2_1(out)
		out = self.cnn2_2(out)
		out = self.cnn3_1(out)
		out = self.cnn3_2(out)
		out = self.cnn3_3(out)
		out = self.cnn4_1(out)
		out = self.cnn4_2(out)
		out = self.cnn4_3(out)
		out = self.cnn5_1(out)
		out = self.cnn5_2(out)
		out = self.cnn5_3(out)
		out = out.reshape(out.size(0), -1)
		out = self.fcn1(out)
		out = self.fcn2(out)
		out = self.out(out)
		return out
	

class VGGModel2(Module):
	def __init__(self, num_classes):
		super(VGGModel2, self).__init__()
		self.model_2 = Sequential(
			Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
			ReLU(),
			Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			ReLU(),
			BatchNorm2d(64),
			MaxPool2d(kernel_size=2, stride=2),
			Dropout(0.25),
			Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			ReLU(),
			BatchNorm2d(128),
			MaxPool2d(kernel_size=2, stride=2),
			Dropout(0.25),
			Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
			ReLU(),
			BatchNorm2d(512),
			MaxPool2d(kernel_size=2, stride=2),
			Dropout(0.25),
			Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			ReLU(),
			BatchNorm2d(512),
			MaxPool2d(kernel_size=2, stride=2),
			Dropout(0.25),
			Flatten(),
			Linear(4608, 256),
			ReLU(),
			BatchNorm1d(256),
			Dropout(0.25),
			Linear(256, 512),
			ReLU(),
			BatchNorm1d(512),
			Dropout(0.25),
			Linear(512, num_classes),
		)

	def forward(self, x):
		return self.model_2(x)
	