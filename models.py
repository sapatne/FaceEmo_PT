from torch.nn import Module, Conv2d, Sequential, BatchNorm2d, ReLU, MaxPool2d, Dropout, Linear

class VGG16(Module):
	def __init__(self, num_classes=7):
		super(VGG16, self).__init__()
		self.cnn1_1 = Sequential(
			Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(64),
			ReLU()
		)
		self.cnn1_2 = Sequential(
			Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
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
			Dropout(0.5),
			Linear(512, 4096),
			ReLU()
		)
		self.fcn2 = Sequential(
			Dropout(0.5),
			Linear(4096, 4096),
			ReLU()
		)
		self.fcn3 = Sequential(
			Dropout(0.5),
			Linear(4096, num_classes),
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
		out = self.fcn3(out)
		return out