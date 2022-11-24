from dataset import DigitDataset
from torch.utils.data import DataLoader
import utils
import torchvision.transforms as transforms
from model import DigitModel
from torch import optim
from tqdm import tqdm
from loss import YoloLoss
import torch
import utils
from PIL import Image

#IMG_DIR = '/Users/cesarolivares/Documents/Maestria CIC/Neuromorficos/GraphNeuroVision/pcnn_emnist'
#LABEL_DIR = '/Users/cesarolivares/Documents/Maestria CIC/Neuromorficos/GraphNeuroVision/labels'
#CSV_DIR = "train.csv"

#IMG_DIR = '/Users/cesarolivares/Documents/Maestria CIC/Neuromorficos/GraphNeuroVision/drone_test_images'
#LABEL_DIR = '/Users/cesarolivares/Documents/Maestria CIC/Neuromorficos/GraphNeuroVision/drone_test_labels'
#CSV_DIR = '/Users/cesarolivares/Documents/Maestria CIC/Neuromorficos/GraphNeuroVision/archive/drone_test.csv'

IMG_DIR = '/Users/cesarolivares/Documents/Maestria CIC/Neuromorficos/GraphNeuroVision/drone_train_images'
LABEL_DIR = '/Users/cesarolivares/Documents/Maestria CIC/Neuromorficos/GraphNeuroVision/drone_train_labels'
CSV_DIR = '/Users/cesarolivares/Documents/Maestria CIC/Neuromorficos/GraphNeuroVision/archive/drone_train.csv'


BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0
DEVICE = 'cpu'

toTensor = transforms.ToTensor()
toResize = transforms.Resize((100,100))
#toResize = transforms.Resize((320,320))

trans = utils.Compose([transforms.ToTensor()])

train_dataset = DigitDataset(csv_file=CSV_DIR,
                            img_dir=IMG_DIR,
                            label_dir=LABEL_DIR,
                            transform=trans)

train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

model = DigitModel()
model.load_state_dict(torch.load('drone_model_trained.pth'))
#model.load_state_dict(torch.load('drone_model_trained.pth'))

pred_boxes, target_boxes = utils.get_bboxes(loader=train_loader, model=model, iou_threshold=0.5,threshold=0.4, device='cpu')

mean_avg_prec = utils.mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )

print(f"Train mAP: {mean_avg_prec}")



"""img = Image.open('/Users/cesarolivares/Documents/Maestria CIC/Neuromorficos/GraphNeuroVision/drone_train_images/338.JPEG')
img_tensor = toTensor(img)
img_tensor_rsh = toResize(img_tensor)
x = torch.unsqueeze(img_tensor_rsh, 0)
bboxes = utils.cellboxes_to_boxes(model(x))
bboxes = utils.non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
utils.plot_image(x[0].permute(1,2,0).to("cpu"), bboxes)
a = 5"""

for x, y in train_loader:
            x = x.to(DEVICE)
            for idx in range(8):
                bboxes = utils.cellboxes_to_boxes(model(x)) #0.8, 0.8
                bboxes = utils.non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
                utils.plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
                a =5

pred_boxes, target_boxes = utils.get_bboxes(loader=train_loader, model=model, iou_threshold=0.5,threshold=0.4, device='cpu')

mean_avg_prec = utils.mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint",
        num_classes=1)

print(f"Train mAP: {mean_avg_prec}")

