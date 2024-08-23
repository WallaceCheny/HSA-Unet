import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from networks.Unet import UNet
from networks.HSAUnet import HSAUnet
import torch
import cv2

# Function to load ISIC images and masks
def load_isic_image(image_path):
    # 加载图像
    image = Image.open(image_path)
    # 转换为RGB格式
    image = image.convert('RGB')
    image = image.resize((224, 224))
    # 转换为numpy数组
    image_array = np.array(image)
    return image_array

def load_isic_mask(mask_path):
    # return (np.array(Image.open(mask_path)) > 127).astype(int)
    # 加载图像
    image = Image.open(mask_path)
    # 转换为RGB格式
    image = image.convert('RGB')
    image = image.resize((224, 224))
    # 转换为numpy数组
    image_array = (np.array(image)  > 127).astype(int)
    return image_array

# Function to save the segmentation result as a PNG file
def save_segmentation_result(segmentation_result, output_path):
    img = Image.fromarray(segmentation_result.astype(np.uint8)*255)
    img.save(output_path)

def save_segmentation(image_path, out_path):
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = UNet(in_ch=3, num_classes=2).to(device)
    # model.load_state_dict(torch.load('./model_out/Unet_Isic_epoch_29.pth', map_location=device))
    model = HSAUnet(num_classes=2).to(device)
    model.load_state_dict(torch.load('./model_out/HSAUnet_Isic_epoch_29.pth', map_location=device))

    model.eval()

    # Load the ISIC image and preprocess it
    image = load_isic_image(image_path)
    image = image / 255.0  # Normalize the image
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(device)  # Convert to torch tensor

    # Perform segmentation
    with torch.no_grad():
        output = model(image)
        output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    save_segmentation_result(output, out_path)
    # # Visualize the result
    # plt.imshow(output, cmap='gray')
    # plt.title('U-Net Segmentation Result')
    # plt.show()

# Function to visualize the ISIC image and segmentation results
def visualize_isic(image, ground_truth, unet_result, swin_unet_result):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image)
    axes[0].set_title('Original Image')

    axes[1].imshow(ground_truth, cmap='gray')
    axes[1].set_title('Ground Truth')

    axes[2].imshow(unet_result, cmap='gray')
    axes[2].set_title('U-Net Result')

    axes[3].imshow(swin_unet_result, cmap='gray')
    axes[3].set_title('Swin-UNet Result')

    plt.show()

# Function to apply color to mask
def apply_color_to_mask(mask, color):
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i in range(3):
        colored_mask[..., i] = (mask == 1) * color[i]
    return colored_mask

def find_boundaries(mask):
    mask = mask * 255
    boundaries = cv2.Canny(mask.astype(np.uint8), 100, 200)
    return boundaries

def overlay_boundaries_on_image(image, boundaries, color):
    overlay = image.copy()
    overlay[boundaries != 0] = color
    return overlay


def visualize_segmentation(input_images, ground_truth_masks, predicted_masks, model_name):
    fig, axes = plt.subplots(3, len(input_images), figsize=(12, 8))

    for i in range(len(input_images)):
        image = load_isic_image(input_images[i])
        ground_truth = load_isic_mask(ground_truth_masks[i])
        predicted = load_isic_mask(predicted_masks[i])

        # Find boundaries
        ground_truth_boundaries = find_boundaries(ground_truth)
        predicted_boundaries = find_boundaries(predicted)

        # Colors for boundaries
        ground_truth_color = [0, 255, 0]  # Green
        predicted_color = [0, 0, 255]  # Blue

        # Overlay boundaries on the original image
        image_with_ground_truth = overlay_boundaries_on_image(image, ground_truth_boundaries, ground_truth_color)
        image_with_predicted = overlay_boundaries_on_image(image_with_ground_truth, predicted_boundaries, predicted_color)

        # Display Input Image
        axes[0, i].imshow(image)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Input Image', fontsize=12)

        # Display Ground Truth Mask
        axes[1, i].imshow(ground_truth, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Ground Truth', fontsize=12)

        # Display Predicted Mask
        axes[2, i].imshow(image_with_predicted)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel(model_name, fontsize=12)

    fig.suptitle('ISIC-2018 Challenge', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
# Function to overlay mask on image
def overlay_mask_on_image(image, mask, color):
    colored_mask = apply_color_to_mask(mask, color)
    overlay = image.copy()
    overlay[mask == 1] = (overlay[mask == 1] * 0.5 + colored_mask[mask == 1] * 0.5).astype(np.uint8)
    return overlay

# Function to visualize the ISIC image, ground truth, and predicted result
def visualize_combined(image, ground_truth, unet_result, swin_unet_result):
    # Define colors for the masks
    ground_truth_color = [255, 0, 0]  # Red
    unet_result_color = [0, 255, 0]   # Green
    swin_unet_result_color = [0, 0, 255]  # Blue

    # Overlay masks on the original image
    image_with_ground_truth = overlay_mask_on_image(image, ground_truth, ground_truth_color)
    image_with_unet_result = overlay_mask_on_image(image, unet_result, unet_result_color)
    image_with_swin_unet_result = overlay_mask_on_image(image, swin_unet_result, swin_unet_result_color)

    # Combine all images into one
    combined_image = np.concatenate((image_with_ground_truth, image_with_unet_result, image_with_swin_unet_result), axis=1)

    # Plot the combined image
    plt.figure(figsize=(20, 10))
    plt.imshow(combined_image)
    plt.axis('off')
    plt.title('Ground Truth (Red) | U-Net Result (Green) | Swin-UNet Result (Blue)')
    plt.show()

if __name__ == '__main__':
    # file_name ='/6.png'
    # image_mask = ['images', 'masks']
    # image_path = f'./data/Isic/isic2018/val/' + image_mask[0] + file_name
    # save_segmentation(image_path)
    # # Paths to the ISIC image, ground truth mask, and model results
    # ground_truth_path = f'./data/Isic/isic2018/val/' + image_mask[1] + file_name
    # unet_result_path = 'path_to_unet_result.png'
    # swin_unet_result_path = 'path_to_swin_unet_result.png'
    #
    # # Load the image and masks
    # image = load_isic_image(image_path)
    # ground_truth = load_isic_mask(ground_truth_path)
    # unet_result = load_isic_mask(unet_result_path)
    # swin_unet_result = load_isic_mask(swin_unet_result_path)
    #
    # visualize_combined(image, ground_truth, unet_result, swin_unet_result)
    # # # Visualize the results
    # # visualize_isic(image, ground_truth, unet_result, swin_unet_result)

    # Paths to the images and masks
    input_images = ['./data/Isic/isic2018/val/images/DAE-03.png', './data/Isic/isic2018/val/images/DAE-02.png']
    ground_truth_masks = ['./data/Isic/isic2018/val/masks/DAE-01.png', './data/Isic/isic2018/val/masks/DAE-02.png']
    predicted_masks = ['DAE-01.png', 'DAE-02.png']
    for i in range(len(input_images)):
        save_segmentation(input_images[i], predicted_masks[i])

    model_name = 'UNet'

    # Visualize the segmentation results
    visualize_segmentation(input_images, ground_truth_masks, predicted_masks, model_name)