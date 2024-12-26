# 1. loading lib
import torch
import h5py
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from networks.SGFormer import SGFormer
from networks.vision_transformer import SwinUnet
from scipy.ndimage.interpolation import zoom
from networks.Unet import UNet
from config import get_config

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
config = get_config(args)

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))

# Normalize images for visualization
def normalize_image(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    return img.astype(np.uint8)

# Generate predictions
def get_prediction(model, image):
    model.eval()
    input_image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    with torch.no_grad():
        outputs = model(input_image)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        predicted_mask = out.cpu().detach().numpy()
    # return predicted_mask
    prediction = predicted_mask # (predicted_mask > 0.5).astype(np.uint8)  # Threshold the predictions
    return prediction
    # [9, 244, 244] -> [244, 244]
    # if(len(prediction.shape) <= 2):
    #     return prediction
    # out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
    # prediction[0, :, :] = 0
    # prediction = np.argmax(prediction, axis=0)
    # return prediction

if __name__ == "__main__":
    # Load the .h5 file
    file_path = './data/Synapse/test_vol_h5/case0035.npy.h5'
    with h5py.File(file_path, 'r') as f:
        images = f['image'][:]  # Assuming 'image' is the key for the images dataset
        labels = f['label'][:]  # Assuming 'label' is the key for the labels dataset

    sample_image = images[67]
    sample_label = labels[67]
    x, y = sample_image.shape
    img_size = 224
    if x != img_size or y != img_size:
        sample_image = zoom(sample_image, (img_size / x, img_size / y), order=3)  # why not 3?
        sample_label = zoom(sample_label, (img_size / x, img_size / y), order=0)

    # Normalize a sample image and label for display
    sample_image = normalize_image(sample_image)  # Take the first image
    sample_label = sample_label  # Take the corresponding label

    # Create dummy models
    model_unet = UNet(num_classes=9)
    model_unet.load_state_dict(torch.load('./model_out/Unet_epoch_29.pth'))
    model_sgformer = SGFormer(num_classes=9)
    model_sgformer.load_state_dict(torch.load('./model_out/synapse_epoch_39.pth'))
    model_swin_unet = SwinUnet(config, img_size=img_size, num_classes=9)
    model_swin_unet.load_state_dict(torch.load('./model_out/swinunet_epoch_99.pth', map_location=torch.device('cpu')))
    # model_transunet = DummyModel()
    model_missformer = DummyModel()


    prediction_sgformer = get_prediction(model_sgformer, sample_image)
    prediction_swin_unet = get_prediction(model_swin_unet, sample_image)
    prediction_unet = get_prediction(model_unet, sample_image)
    prediction_missformer = get_prediction(model_missformer, sample_image)


    # Define color mappings for different organs (same as in the provided example)
    color_mapping = {
        0: [0, 0, 0],  # Background (black)
        1: [0, 0, 255],  # Aorta (blue)
        2: [0, 255, 0],  # Gallbladder (green)
        3: [255, 0, 0],  # Left Kidney (red)
        4: [0, 255, 255],  # Right Kidney (cyan)
        5: [255, 0, 255],  # Liver (magenta)
        6: [255, 255, 0],  # Pancreas (yellow)
        7: [0, 191, 255],  # Spleen (deep sky blue)
        8: [255, 255, 255]  # Stomach (white)
    }


    def apply_color_mask(image, mask, color_mapping):
        color_image = np.zeros((*image.shape, 3), dtype=np.uint8)
        # color_image = np.array(image)
        # color_image = np.repeat(np.expand_dims(color_image, axis=-1), 3, axis=-1)
        for label, color in color_mapping.items():
            color_image[mask == label] = color
        return color_image


    # Apply color masks
    color_mask_gt = apply_color_mask(sample_image, sample_label, color_mapping)
    color_mask_sgformer = apply_color_mask(sample_image, prediction_swin_unet, color_mapping)
    color_mask_transunet = apply_color_mask(sample_image, prediction_unet, color_mapping)
    color_mask_missformer = apply_color_mask(sample_image, prediction_missformer, color_mapping)

    # Create the composite image layout
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    # Titles for each column
    titles = ["Image", "Ground Truth", "Swin-Unet", "Unet", "MISSFormer"]

    # Plot images
    axes[0, 0].imshow(sample_image, cmap='gray')
    axes[0, 0].set_title(titles[0])
    axes[0, 0].axis('off')

    axes[0, 1].imshow(color_mask_gt)
    axes[0, 1].set_title(titles[1])
    axes[0, 1].axis('off')

    axes[0, 2].imshow(color_mask_sgformer)
    axes[0, 2].set_title(titles[2])
    axes[0, 2].axis('off')

    axes[0, 3].imshow(color_mask_transunet)
    axes[0, 3].set_title(titles[3])
    axes[0, 3].axis('off')

    axes[0, 4].imshow(color_mask_missformer)
    axes[0, 4].set_title(titles[4])
    axes[0, 4].axis('off')

    # Repeat for additional slices if desired
    # For example, using slices 1 and 2
    slices = [70, 75]
    for i, s in enumerate(slices):
        sample_image = normalize_image(images[s])
        sample_label = labels[s]
        x, y = sample_image.shape
        if x != img_size or y != img_size:
            sample_image = zoom(sample_image, (img_size / x, img_size / y), order=3)  # why not 3?
            sample_label = zoom(sample_label, (img_size / x, img_size / y), order=0)
        prediction_sgformer = get_prediction(model_sgformer, sample_image)
        prediction_unet = get_prediction(model_unet, sample_image)
        prediction_missformer = get_prediction(model_missformer, sample_image)

        color_mask_gt = apply_color_mask(sample_image, sample_label, color_mapping)
        color_mask_sgformer = apply_color_mask(sample_image, prediction_sgformer, color_mapping)
        color_mask_transunet = apply_color_mask(sample_image, prediction_unet, color_mapping)
        color_mask_missformer = apply_color_mask(sample_image, prediction_missformer, color_mapping)

        axes[i + 1, 0].imshow(sample_image, cmap='gray')
        axes[i + 1, 0].axis('off')

        axes[i + 1, 1].imshow(color_mask_gt)
        axes[i + 1, 1].axis('off')

        axes[i + 1, 2].imshow(color_mask_sgformer)
        axes[i + 1, 2].axis('off')

        axes[i + 1, 3].imshow(color_mask_transunet)
        axes[i + 1, 3].axis('off')

        axes[i + 1, 4].imshow(color_mask_missformer)
        axes[i + 1, 4].axis('off')

    # Adjust spacing
    plt.tight_layout()
    plt.show()

    # # 2. loading image and pretrain model
    # # Load the original image
    # image_path = 'path_to_your_image.png'
    # original_image = Image.open(image_path).convert('RGB')
    #
    # # Define the model and load the pretrained weights
    # model_path = 'path_to_your_trained_model.pth'
    # model = UNet(in_ch=3, num_classes=1)  # Adjust as per your model definition
    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    #
    #
    # # 3. preprocessing the image
    # # Define image transformations
    # preprocess = transforms.Compose([
    #     transforms.Resize((128, 128)),  # Adjust size accordingly
    #     transforms.ToTensor(),
    # ])
    #
    # input_image = preprocess(original_image).unsqueeze(0)  # Add batch dimension
    #
    # # 4. generating the process and segmentation mask
    # # Move the model and input image to the appropriate device (CPU or GPU)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    # input_image = input_image.to(device)
    #
    # # Predict the segmentation mask
    # with torch.no_grad():
    #     output = model(input_image)
    # predicted_mask = torch.sigmoid(output).squeeze().cpu().numpy()  # Assuming sigmoid activation for binary segmentation
    # predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Thresholding
    #
    # # 5. overlaying the segmentation Mask on the original image
    # # Convert the original image to numpy array
    # original_image_np = np.array(original_image)
    #
    # # Resize the predicted mask to the original image size
    # predicted_mask_resized = cv2.resize(predicted_mask, (original_image_np.shape[1], original_image_np.shape[0]))
    #
    # # Create an overlay image with color for the mask
    # overlay_image = original_image_np.copy()
    # overlay_image[predicted_mask_resized == 1] = [255, 0, 0]  # Red color for the mask
    #
    # # Blend the original image and the overlay image
    # alpha = 0.5  # Transparency factor
    # blended_image = cv2.addWeighted(original_image_np, alpha, overlay_image, 1 - alpha, 0)
    #
    # # 6. displaying the results
    # # Plot the images using matplotlib
    # plt.figure(figsize=(10, 10))
    #
    # plt.subplot(1, 3, 1)
    # plt.title('Original Image')
    # plt.imshow(original_image_np)
    # plt.axis('off')
    #
    # plt.subplot(1, 3, 2)
    # plt.title('Predicted Mask')
    # plt.imshow(predicted_mask_resized, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplot(1, 3, 3)
    # plt.title('Overlay Image')
    # plt.imshow(blended_image)
    # plt.axis('off')
    #
    # plt.show()
