from torch.utils.data import DataLoader
from customdataset import train_dataset, test_dataset, val_dataset
from torchvision import utils
import matplotlib.pyplot as plt


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Helper function to show a batch
def show_labels_batch(sample_batched):
    """Show image with labels for a batch of samples."""
    images_batch, labels_batch = \
            sample_batched['image'], sample_batched['label']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['label'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_labels_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break


print(train_dataset[3]['image'].dtype)