import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from models.MIMO import MIMO
from models.SSRNET import SSRNET
from models.SingleCNN import SpatCNN, SpecCNN
from models.SSFCNN import SSFCNN, ConSSFCNN
from models.MCT_rectangle import MCT_rectangle
from utils import *
from data_loader import build_datasets
from validate import validate
from train import train, train_contrast
import pdb
import args_parser
from torch.nn import functional as F
from models.moco import Moco

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(args)


def main():

    # Custom dataloader
    train_list, test_list = build_datasets(args.root,
                                           args.dataset,
                                           args.image_size,
                                           args.n_select_bands,
                                           args.scale_ratio)
    if args.dataset == 'PaviaU':
        args.n_bands = 103
    elif args.dataset == 'paviaC' or args.dataset == 'Pavia':
        args.n_bands = 102
    elif args.dataset == 'Botswana':
        args.n_bands = 145
    elif args.dataset == 'KSC':
        args.n_bands = 176
    elif args.dataset == 'Urban':
        args.n_bands = 162
    elif args.dataset == 'IndianP':
        args.n_bands = 200
    elif args.dataset == 'DC' or args.dataset == 'Washington':
        args.n_bands = 191
    elif args.dataset == 'MUUFL_HSI':
        args.n_bands = 64
    elif args.dataset == 'salinas_corrected':
        args.n_bands = 204
    elif args.dataset == 'Houston_HSI' or args.dataset == 'Houston13':
        args.n_bands = 144
    elif args.dataset == 'Chikusei':
        args.n_bands = 128
    elif args.dataset == 'Augsburg':
        args.n_bands = 101
    elif args.dataset == 'Cave21':
        args.n_bands = 31

    # Build the models
    if args.arch == 'SSFCNN':
        model = SSFCNN(args.scale_ratio,
                       args.n_select_bands,
                       args.n_bands).cuda()
    elif args.arch == 'SSRNET' or args.arch == 'SpatRNET' or args.arch == 'SpecRNET':
        model = SSRNET(args.arch,
                       args.scale_ratio,
                       args.n_select_bands,
                       args.n_bands,
                       ).cuda()
    elif args.arch == 'SpatCNN':
        model = SpatCNN(args.scale_ratio,
                        args.n_select_bands,
                        args.n_bands).cuda()
    elif args.arch == 'SpecCNN':
        model = SpecCNN(args.scale_ratio,
                        args.n_select_bands,
                        args.n_bands).cuda()

    elif args.arch in ('DRTnet', 'DRTnet_GSIS', 'no_contrast'):
        model = MCT_rectangle(args.arch,
                              args.scale_ratio,
                              args.n_select_bands,
                              args.n_bands,
                              args.dataset,
                              ).cuda()

    elif args.arch == 'MIMO':
        model = MIMO(args.n_select_bands, args.n_bands).cuda()
    else:
        raise ValueError('Unsupported architecture: {}'.format(args.arch))

    image_size = args.image_size
    scale_ratio = args.scale_ratio
    train_ref, train_lr, train_hr = train_list

    h, w = train_ref.size(2), train_ref.size(3)
    h_str = random.randint(0, h - image_size - 1)
    w_str = random.randint(0, w - image_size - 1)

    train_ref = train_ref[:, :, h_str:h_str + image_size, w_str:w_str + image_size]
    train_lr = F.interpolate(train_ref, scale_factor=1 / (scale_ratio * 1.0))
    train_hr = train_hr[:, :, h_str:h_str + image_size, w_str:w_str + image_size]

    image_lr = to_var(train_lr).detach()
    image_hr = to_var(train_hr).detach()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    parameter_nums = sum(p.numel() for p in model.parameters())
    print("Model size:", str(float(parameter_nums / 1e6)) + 'M')
    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
        .replace('arch', args.arch)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print('Load the chekpoint of {}'.format(model_path))
        recent_psnr = validate(test_list,
                               args.arch,
                               model,
                               0,
                               args.n_epochs)
        print('psnr: ', recent_psnr)

    # Loss and Optimizer
    criterion = nn.MSELoss().cuda()
    contrast_loss = None

    best_psnr = 0
    best_psnr = validate(test_list,
                         args.arch,
                         model,
                         0,
                         args.n_epochs)
    print('psnr: ', best_psnr)

    # Epochs
    print('Start Training: ')
    best_epoch = 0

    for epoch in range(args.n_epochs):
        print('Train_Epoch_{}: '.format(epoch))
        E = Moco(base_encoder=model, image_lr=image_lr, image_hr=image_hr).cuda()
        train_contrast(train_list,
                       args.image_size,
                       args.scale_ratio,
                       args.n_bands,
                       args.arch,
                       model,
                       E,
                       optimizer,
                       criterion,
                       contrast_loss,
                       epoch,
                       args.n_epochs)

        print('Val_Epoch_{}: '.format(epoch))
        recent_psnr = validate(test_list,
                               args.arch,
                               model,
                               epoch,
                               args.n_epochs)
        print('psnr: ', recent_psnr)

        is_best = recent_psnr > best_psnr
        best_psnr = max(recent_psnr, best_psnr)
        if is_best:
            best_epoch = epoch
            if best_psnr > 0:
                torch.save(model.state_dict(), model_path)
                print('Saved!')
                print('')
        print('best psnr:', best_psnr, 'at epoch:', best_epoch)

    print('best_psnr: ', best_psnr)


if __name__ == '__main__':
    main()
