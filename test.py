import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_stomach import Stomach_dataset
from datasets.dataset_isic import Isic_dataset
from networks.DAEFormer import DAEFormer
# from networks.SGFormer import SGFormer
from networks.HSAUnet import HSAUnet
from networks.Unet import UNet
from trainer import trainer_synapse
from utils import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument(
    "--volume_path",
    type=str,
    default="/images/PublicDataset/Transunet_synaps/project_TransUNet/data/Synapse/",
    help="root dir for validation volume data",
)  # for acdc volume_path=root_dir
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument("--num_classes", type=int, default=9, help="output channel of network")
parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse", help="list dir")
parser.add_argument("--output_dir", type=str, default="./model_out", help="output dir")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
parser.add_argument("--max_epochs", type=int, default=400, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
parser.add_argument("--is_savenii", action="store_true", help="whether to save results during inference")
parser.add_argument("--test_save_dir", type=str, default="../predictions", help="saving prediction as nii!")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation network learning rate")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs="+",
)
parser.add_argument("--zip", action="store_true", help="use zipped dataset instead of folder dataset")
parser.add_argument(
    "--cache-mode",
    type=str,
    default="part",
    choices=["no", "full", "part"],
    help="no: no cache, "
    "full: cache all data, "
    "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
)
parser.add_argument("--resume", help="resume from checkpoint")
parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
parser.add_argument(
    "--use-checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory"
)
parser.add_argument(
    "--amp-opt-level",
    type=str,
    default="O1",
    choices=["O0", "O1", "O2"],
    help="mixed precision opt level, if O0, no amp is used",
)
parser.add_argument("--tag", help="tag of experiment")
parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
parser.add_argument("--throughput", action="store_true", help="Test throughput only")

args = parser.parse_args()
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
# config = get_config(args)


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="val", img_size=args.img_size, list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch["case_name"][0]
        metric_i = test_single_volume(
            image,
            label,
            model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=args.z_spacing,
        )
        metric_list += np.array(metric_i)
        # it log too much
        if i_batch % 10 == 0:
            logging.info(
                "idx %d case %s mean_dice %f mean_hd95 %f"
                % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1])
            )
    metric_list = metric_list / len(db_test)
    for i in range(0, args.num_classes):
        logging.info("Mean class %d mean_dice %f mean_hd95 %f SE %f, SP %f, ACC %f " % (i, metric_list[i][0], metric_list[i][1],
                                                                                        metric_list[i][2], metric_list[i][3], metric_list[i][4]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    SE = np.mean(metric_list, axis=0)[2]
    SP = np.mean(metric_list, axis=0)[3]
    ACC = np.mean(metric_list, axis=0)[4]
    logging.info("Testing performance in best val model: mean_dice : %f mean_hd95 : %f, SE: %f, SP: %f, ACC: %f " % (performance, mean_hd95, SE, SP, ACC))
    return "Testing Finished!"


if __name__ == "__main__":
    # from sklearn.model_selection import train_test_split
    # from sklearn.datasets import load_breast_cancer
    # import pandas as pd
    #
    # cancer = load_breast_cancer()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        "Synapse": {
            "Dataset": Synapse_dataset,
            "z_spacing": 1,
        },
        "Stomach":{
            "Dataset":Stomach_dataset,
            "z_spacing": 1,
        },
        "Isic":{
            "Dataset":Isic_dataset,
            "z_spacing": 1,
        }
    }
    dataset_name = args.dataset
    args.Dataset = dataset_config[dataset_name]["Dataset"]
    args.z_spacing = dataset_config[dataset_name]["z_spacing"]
    args.is_pretrain = True

    # net = DAEFormer(num_classes=args.num_classes).cuda(0) # SGFormer(num_classes=args.num_classes).cuda(0) #
    # net = UNet(num_classes=2).cuda(0)
    net = HSAUnet(num_classes=2).cuda(0)
    # snapshot = os.path.join(args.output_dir, "best_model.pth")
    # snapshot = os.path.join(args.output_dir, "isic_epoch_79.pth")
    # snapshot = os.path.join(args.output_dir, "HSAUnet_Isic_epoch_29.pth")
    # if not os.path.exists(snapshot):
    #     snapshot = snapshot.replace("best_model", "synapse_epoch_" + str(args.max_epochs - 1))
    # msg = net.load_state_dict(torch.load(snapshot))
    # msg = net.load_state_dict(torch.load('./model_out/DAEFormer_Isic_epoch_29.pth'))
    msg = net.load_state_dict(torch.load('./model_out/HSAUent_Isic_epoch_29.pth'))
    print("self trained HSAUnet", msg)
    snapshot_name = "HSAUnet_Normal_Loss_ISIC2018" # snapshot.split("/")[-1]

    log_folder = "./test_log"
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=log_folder + "/" + snapshot_name + ".txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)
