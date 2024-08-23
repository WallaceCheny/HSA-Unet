import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from torch.nn import functional as F
from torchvision import transforms
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes):
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2] + 2, target.shape[-1] + 2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(),
                                      padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha,
                    0.8)  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes

class BoundaryLoss(nn.Module):
    # def __init__(self, **kwargs):
    def __init__(self, classes) -> None:
        super().__init__()
        # # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        # self.idc: List[int] = kwargs["idc"]
        self.idx = [i for i in range(classes)]

    def compute_sdf1_1(self, img_gt, out_shape):
        """
        compute the normalized signed distance map of binary mask
        input: segmentation, shape = (batch_size, x, y, z)
        output: the Signed Distance Map (SDM)
        sdf(x) = 0; x in segmentation boundary
                -inf|x-y|; x in segmentation
                +inf|x-y|; x out of segmentation
        normalize sdf to [-1, 1]
        """
        img_gt = img_gt.cpu().numpy()
        img_gt = img_gt.astype(np.uint8)

        normalized_sdf = np.zeros(out_shape)

        for b in range(out_shape[0]): # batch size
                # ignore background
            for c in range(1, out_shape[1]):
                posmask = img_gt[b].astype(np.bool)
                if posmask.any():
                    negmask = ~posmask
                    posdis = distance(posmask)
                    negdis = distance(negmask)
                    boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                    sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                    sdf[boundary==1] = 0
                    normalized_sdf[b][c] = sdf

        return normalized_sdf

    def forward(self, outputs, gt):
        """
        compute boundary loss for binary segmentation
        input: outputs_soft: sigmoid results,  shape=(b,2,x,y,z)
            gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
        output: boundary_loss; sclar
        """
        outputs_soft = F.softmax(outputs, dim=1)
        gt_sdf = self.compute_sdf1_1(gt, outputs_soft.shape)
        pc = outputs_soft[:,self.idx,...]
        dc = torch.from_numpy(gt_sdf[:,self.idx,...]).cuda()
        multipled = torch.einsum('bxyz, bxyz->bxyz', pc, dc)
        bd_loss = multipled.mean()

        return bd_loss

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        # # 计算混淆矩阵
        # pred: [224, 224], gt: [224, 224]
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        # cm = confusion_matrix(pred_flat, gt_flat)
        # se1 = cm[1][1] / (cm[1][0] + cm[1][1])
        # sp1 = cm[0][0] / (cm[0][1] + cm[0][0])
        # acc1 = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1])
        # Compute the confusion matrix
        cm = confusion_matrix(gt_flat, pred_flat)
        # Extract TN, FP, FN, TP from the confusion matrix
        TN, FP, FN, TP = cm.ravel()
        # Calculate Precision
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        # Calculate Accuracy
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        # Calculate Sensitivity (Recall)
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        # pred = np.array(pred, dtype=bool)
        # gt = np.array(gt, dtype=bool)
        se2 = metric.sensitivity(pred_flat, gt_flat)
        sp2 = metric.specificity(pred_flat, gt_flat)
        # acc2 = accuracy_score(gt, pred)
        # assert acc1 == acc2
        return dice, hd95, sensitivity, specificity, accuracy
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 0, 0, 0
    else:
        return 0, 0, 0, 0, 0

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        if len(image.shape) != len(label.shape): # meaning that is h, w, c
            prediction = np.zeros_like(label)
            x, y = image.shape[0], image.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                image = zoom(image, (patch_size[0] / x, patch_size[1] / y, 1), order=3)  # previous using 0
            x_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            input = x_transforms(image).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                # outputs = F.interpolate(outputs, size=slice.shape[:], mode='bilinear', align_corners=False)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction = pred
        else:
            prediction = np.zeros_like(label)
            for ind in range(image.shape[0]):
                slice = image[ind, :, :]
                x, y = slice.shape[0], slice.shape[1]
                if x != patch_size[0] or y != patch_size[1]:
                    slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
                x_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ])
                input = x_transforms(slice).unsqueeze(0).float().cuda()

                net.eval()
                with torch.no_grad():
                    outputs = net(input)
                    # outputs = F.interpolate(outputs, size=slice.shape[:], mode='bilinear', align_corners=False)
                    out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                    out = out.cpu().detach().numpy()
                    if x != patch_size[0] or y != patch_size[1]:
                        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                    else:
                        pred = out
                    prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(0, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

if __name__ == "__main__":
    # # 真实标签和预测标签
    # y_true = [0, 1, 1, 0, 1, 0]
    # y_pred = [0, 1, 0, 0, 1, 1]
    #
    # # 计算混淆矩阵
    # cm = confusion_matrix(y_true, y_pred)
    #
    # # 计算 DSC、SE、SP 和 ACC
    # dsc = (2 * cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1])
    # se = cm[1][1] / (cm[1][0] + cm[1][1])
    # sp = cm[0][0] / (cm[0][1] + cm[0][0])
    # acc = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1])
    #
    # print("DSC:", dsc)
    # print("SE:", se)
    # print("SP:", sp)
    # print("ACC:", acc)

    y_true = np.array([0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6])

    y_pred_binary = np.where(y_pred > 0.5, 1, 0)

    dsc, hd, se, sp, acc = calculate_metric_percase(y_true == 1, y_pred_binary == 1)

    print("DSC:", dsc)
    print("hd:", hd)
    print("SE:", se)
    print("SP:", sp)
    print("ACC:", acc)
    # # SE
    # sensitivity = metric.sensitivity(y_true == 1, y_pred_binary == 1)
    # # SP
    # specificity = metric.specificity(y_true == 1, y_pred_binary == 1)
    # # ACC
    # accuracy = accuracy_score(y_true, y_pred_binary)
