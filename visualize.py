import os.path
import torch
from networks.SGFormer import SGFormer
from networks.SCUNet import SwinTransformerSys
from networks.DETFormer import DTEFormer
from networks.Unet import UNet
import argparse
import numpy as np
import monai
import torchvision.transforms as transforms
from PIL import Image
import h5py
import cv2
from datasets.dataset_synapse import Synapse_dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="data/Synapse/train_npz",
    help="root dir for train data",
)
parser.add_argument("--num_classes", type=int, default=9, help="output channel of network")
parser.add_argument("--output_dir", type=str, default="./model_out", help="output dir")
parser.add_argument("--test_vol", type=str, default="data/Synapse/test_vol_h5", help="output dir")
parser.add_argument("--test_path", type=str, default="data/Synapse/test_vol_h5", help="output dir")
parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse", help="list dir")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
args = parser.parse_args()


def get_image_label(file_path):
    data = h5py.File(file_path)
    img, label = data['image'][:], data['label'][:]
    sample = {'image': img, 'label': label}
    return sample


def transform_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0)
    return image


def visual_save(original_img, pred, save_path, save_name):
    blue = [30, 144, 255]  # aorta
    green = [0, 255, 0]  # gallbladder
    red = [255, 0, 0]  # left kidney
    cyan = [0, 255, 255]  # right kidney
    pink = [255, 0, 255]  # liver
    yellow = [255, 255, 0]  # pancreas
    purple = [128, 0, 255]  # spleen
    orange = [255, 128, 0]  # stomach
    label2color_dict = {
        1: blue,  # aorta
        2: green,
        3: red,  # left kidney
        4: cyan,  # right kidney
        5: pink,  # liver
        6: yellow,  # pancreas
        7: purple,  # spleen
        8: orange,  # stomach
    }
    original_img = original_img * 255.0
    original_img = original_img.astype(np.uint8)
    pred = pred.astype(np.uint8)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    for i in range(1, 9):
        original_img = np.where(pred == i, np.full_like(original_img, label2color_dict[i]), original_img)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(save_path, save_name), original_img)


if __name__ == "__main__":
    net = SGFormer(num_classes=args.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)
    if device.type == "cuda":
        net = net.cuda(0)

    snapshot = os.path.join(args.output_dir, "synapse_epoch_39.pth")
    msg = net.load_state_dict(torch.load(snapshot))
    print('loaded the net dict...')

    # 切换推理模式
    net.eval()
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.ToTensor()
    db_test = Synapse_dataset(base_dir=args.root_path, split="train", list_dir=args.list_dir,
                              img_size=args.img_size, norm_x_transform=x_transforms, norm_y_transform=y_transforms)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(test_loader):
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name']
            output = net(image)
            _image = image[0]
            # _out_image = output[0]
            # _image = torch.stack([_image, _out_image], dim=0)
            save_image(_image, 'test.png')

