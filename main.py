import os
import argparse
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
from sklearn.metrics import jaccard_similarity_score, f1_score

import utils
#import preprocess
from loader import loader
from models.dense_net import DenseNet

from trainers.ClassifyTrainer import ClassifyTrainer

def arg_parse():
    desc = "Osteo Classification"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default="0,1,2,3",
                        help="Select GPU Numbering | 0,1,2,3 | ")
    parser.add_argument('--cpus', type=int, default="4",
                        help="Select CPU Number workers")
    parser.add_argument('--model', type=str, default='dense_net',
                        choices=["dense_net"], required=True)
    parser.add_argument('--norm', type=str, default='bn', choices=["in", "bn", "bin"])
    parser.add_argument('--act', type=str, default='relu', choices=["relu", "elu", "leaky", "prelu"])

    parser.add_argument('--augment', type=str, default='',
                        help='The type of augmentaed ex) crop,rotate ..  | crop | flip | elastic | rotate |')

    # TODO : Weighted BCE
    parser.add_argument('--loss', type=str, default='BCE',
                        choices=['BCE'])
    # Loss Params
    parser.add_argument('--focal_gamma', type=float, default='2', help='')
    parser.add_argument('--t_alpha', type=float, default='0.3', help='')

    parser.add_argument('--dtype', type=str, default='float',
                        choices=['float', 'half'],
                        help='The torch dtype | float | half |')

    parser.add_argument('--fold', type=str, default='')

    parser.add_argument('--sampler', type=str, default='',
                        choices=['weight', ''],
                        help='The setting sampler')

    parser.add_argument('--epoch', type=int, default=300, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='The size of batch')
    parser.add_argument('--test', action="store_true", help='The size of batch')

    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory name to save the model')

    # Adam Parameter
    parser.add_argument('--lrG',   type=float, default=0.0005)
    parser.add_argument('--beta',  nargs="*", type=float, default=(0.5, 0.999))

    return parser.parse_args()


def arg_check(arg):
    if len(arg.gpus) <= 0:
        raise argparse.ArgumentTypeError("gpus must be 0,1,2 or 2,3,4 ...")
    for chk in arg.gpus:
        if chk not in "0123456789,":
            raise argparse.ArgumentTypeError("gpus must be 0,1,2 or 2,3,4 ...")

    check_dict = [("cpus", arg.cpus), ("epoch", arg.epoch), ("batch", arg.batch_size), ("lrG", arg.lrG)]
    for chk in check_dict:
        if chk[1] <= 0:
            raise argparse.ArgumentTypeError("%s <= 0" % (chk[0]))
    if arg.beta[0] <= 0 or arg.beta[1] <= 0:
        raise argparse.ArgumentTypeError("betas <= 0")


if __name__ == "__main__":
	arg = arg_parse()
	arg_check(arg)

	os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
	torch_device = torch.device("cuda")

	train_path = "data/train/"
	val_path = "data/val/"
	test_path = "data/test/"

	train_loader = loader(train_path, arg.batch_size, transform = None, sampler = '',
		torch_type = 'float', cpus = 4, shuffle = True, drop_last = True)
	val_loader = loader(val_path, arg.batch_size, transform = None, sampler = '',
		torch_type = 'float', cpus = 4, shuffle = False, drop_last = False)
	test_loader = loader(test_path, arg.batch_size, transform = None, sampler = '',
		torch_type = 'float', cpus = 4, shuffle = False, drop_last = False)

	net = DenseNet(growthRate = 12, depth = 40, reduction = 0.5, bottleneck = True, nClasses = 2)

	net = nn.DataParallel(net).to(torch_device)

	# TODO : redfine loss
	class_loss = nn.BCEWithLogitsLoss()

	model = ClassifyTrainer(arg, net, torch_device, class_loss = class_loss)

	if arg.test is False:
		model.train(train_loader, val_loader)
