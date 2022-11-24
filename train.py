from dataset import DigitDataset
from torch.utils.data import DataLoader
import utils
import torchvision.transforms as transforms
from model import DigitModel
from torch import optim
from tqdm import tqdm
from loss import YoloLoss
import torch

#IMG_DIR = '/Users/cesarolivares/Documents/Maestria CIC/Neuromorficos/GraphNeuroVision/pcnn_images'
#LABEL_DIR = '/Users/cesarolivares/Documents/Maestria CIC/Neuromorficos/GraphNeuroVision/labels'
#CSV_DIR = 'train.csv'

IMG_DIR = '/Users/cesarolivares/Documents/Maestria CIC/Neuromorficos/GraphNeuroVision/drone_train_images'
LABEL_DIR = '/Users/cesarolivares/Documents/Maestria CIC/Neuromorficos/GraphNeuroVision/drone_train_labels'
CSV_DIR = '/Users/cesarolivares/Documents/Maestria CIC/Neuromorficos/GraphNeuroVision/archive/drone_train.csv'

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0
DEVICE = 'cpu'


trans = utils.Compose([transforms.ToTensor()])

train_dataset = DigitDataset(csv_file=CSV_DIR,
                            img_dir=IMG_DIR,
                            label_dir=LABEL_DIR,
                            transform=trans)

train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

model = DigitModel() 

optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

loss_fn = YoloLoss(S=7, B=2, C=1)

for epoch in range(EPOCHS):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

torch.save(model.state_dict(), 'drone_model_trained.pth')

    