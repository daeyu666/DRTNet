import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
# from models.MCT import MCT
# from models.MCT_RCB import MCT_RCB ######### # 3.20
from models.MCT_rectangle import MCT_rectangle ######3.21
from models.SSRNET import SSRNET
from models.SingleCNN import SpatCNN, SpecCNN
from models.TFNet import TFNet, ResTFNet
from models.SSFCNN import SSFCNN, ConSSFCNN
from models.MSDCNN import MSDCNN
from utils import *
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam, mrae
from data_loader import build_datasets
# from ablation import ablation
from models.DRTnet import DRTnet
import pdb
import scipy.io as io
import args_parser
from torch.nn import functional as F
import cv2
from time import *
from thop import profile  # 统计模型的 FLOPs 和参数量
args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(args)




def main():  # # 数据集的光谱波段数
    if args.dataset == 'PaviaU':
        args.n_bands = 103
    elif args.dataset == 'paviaC':
        args.n_bands = 102
    elif args.dataset == 'Botswana':
        args.n_bands = 145
    elif args.dataset == 'KSC':
        args.n_bands = 176
    elif args.dataset == 'Urban':
        args.n_bands = 162
    elif args.dataset == 'IndianP':
        args.n_bands = 200
    elif args.dataset == 'Washington':
        args.n_bands = 191
    elif args.dataset == 'DC':
        args.n_bands = 191
    elif args.dataset == 'MUUFL_HSI':
        args.n_bands = 64
    elif args.dataset == 'Houston_HSI':
        args.n_bands = 144
    elif args.dataset == 'salinas_corrected':
        args.n_bands = 204
    elif args.dataset == 'Chikusei':
        args.n_bands = 128
    elif args.dataset == 'Augsburg':
        args.n_bands = 101
    elif args.dataset == 'Cave21':
        args.n_bands = 31
    # Custom dataloader
    train_list, test_list = build_datasets(args.root,
                                           args.dataset,
                                           args.image_size,
                                           args.n_select_bands,
                                           args.scale_ratio)

    # Build the models
    if args.arch == 'SSFCNN':
        model = SSFCNN(args.scale_ratio,
                       args.n_select_bands,
                       args.n_bands)
    if args.arch == 'MCT':
        model = MCT(args.arch,
                    args.scale_ratio,
                    args.n_select_bands,
                    args.n_bands,
                    args.dataset)
    elif args.arch == 'MCT':    # 3.20 窗口
        model = MCT_RCB(args.arch,
                    args.scale_ratio,
                    args.n_select_bands,
                    args.n_bands,
                    args.dataset)
    # elif args.arch == 'DRT-Net' or 'TSTfirst' or 'no_contrast' or 'DRT_old':  # 3.21
    #     model = MCT_rectangle(args.arch,
    #                            args.scale_ratio,
    #                            args.n_select_bands,
    #                            args.n_bands,
    #                           args.n_subs,
    #                           args.n_ovls,
    #                            args.dataset,
    #                           ).cuda()
    # elif args.arch == 'TSTfirst' or 'window' or 'DRT_old':    # 3.21 矩形窗口
    #     model = MCT_rectangle(args.arch,
    #                 args.scale_ratio,
    #                 args.n_select_bands,
    #                 args.n_bands,
    #                           args.n_subs,
    #                           args.n_ovls,
    #                 args.dataset,).cuda()
    elif args.arch == 'ConSSFCNN':
        model = ConSSFCNN(args.scale_ratio,
                          args.n_select_bands,
                          args.n_bands)
    elif args.arch == 'TFNet':
        model = TFNet(args.scale_ratio,
                      args.n_select_bands,
                      args.n_bands)
    elif args.arch == 'ResTFNet':
        model = ResTFNet(args.scale_ratio,
                         args.n_select_bands,
                         args.n_bands)
    elif args.arch == 'MSDCNN':
        model = MSDCNN(args.scale_ratio,
                       args.n_select_bands,
                       args.n_bands)
    elif args.arch == 'SSRNET' or args.arch == 'SpatRNET' or args.arch == 'SpecRNET':
        model = SSRNET(args.arch,
                       args.scale_ratio,
                       args.n_select_bands,
                       args.n_bands)
    elif args.arch == 'SpatCNN':
        model = SpatCNN(args.scale_ratio,
                        args.n_select_bands,
                        args.n_bands)
    elif args.arch == 'SpecCNN':
        model = SpecCNN(args.scale_ratio,
                        args.n_select_bands,
                        args.n_bands)
    elif args.arch == 'DRTnet_GSIS':
        model = MCT_rectangle(args.arch,
                       args.scale_ratio,
                       args.n_select_bands,
                       args.n_bands,
                       args.dataset,
                       ).cuda()
    # elif args.arch == 'DRT_old':
    #     model = DRTnet(args.arch,
    #                    args.scale_ratio,
    #                    args.n_select_bands,
    #                    args.n_bands,
    #                    args.dataset,
    #                    ).cuda()


    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
        .replace('arch', args.arch)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print('Load the chekpoint of {}'.format(model_path))


    test_ref, test_lr, test_hr = test_list
    model.eval()

    # Set mini-batch dataset
    ref = test_ref.float().detach()
    lr = test_lr.float().detach()
    hr = test_hr.float().detach()

    begin_time = time()
    if args.arch == 'SSRNET':
        out, _, _, _, _, _ = model(lr, hr)
    elif args.arch == 'SpatRNET':
        _, out, _, _, _, _ = model(lr, hr)
    elif args.arch == 'SpecRNET':
        _, _, out, _, _, _ = model(lr, hr)
    else:
        # out, _, _, _, _, _ = model(lr, hr)
        out, _,  = model(lr.cuda(), hr.cuda())  # .cuda()
    end_time = time()
    run_time = (end_time - begin_time) * 1000

    # flops
    print()
    flops, params = profile(model, inputs=(lr.cuda(), hr.cuda(),))
    average_times = 1000
    print('Dataset:   {}'.format(args.dataset))
    print('Arch:   {}'.format(args.arch))
    print('params(M),', params / 1e6)
    print('flops(G),', flops / 1e9)

    model.cuda()
    lr = lr.cuda()
    hr = hr.cuda()
    a_time = time()
    for i in range(average_times):
        # out, _, _, _, _ , _= model(lr, hr)
        out, _, = model(lr, hr)
    b_time = time()
    print('average times', average_times)
    print('test time(ms)', 1000 * (b_time - a_time) / average_times)
    print()

    print()
    print()
    print('Dataset:   {}'.format(args.dataset))
    print('Arch:   {}'.format(args.arch))
    print()

    ref = ref.detach().cpu().numpy()
    out = out.detach().cpu().numpy()
    print('ref',ref.shape)
##########################评价指标
    psnr = calc_psnr(ref, out)
    rmse = calc_rmse(ref, out)
    ergas = calc_ergas(ref, out)
    sam = calc_sam(ref, out)
    MRAE = mrae(out,ref)
    print('RMSE:   {:.4f};'.format(rmse))
    print('PSNR:   {:.4f};'.format(psnr))
    print('ERGAS:   {:.4f};'.format(ergas))
    print('SAM:   {:.4f};'.format(sam))
    print('MRAE:   {:.4f};'.format(MRAE))

    # bands order
    if args.dataset == 'Botswana':
        red = 47
        green = 14
        blue = 3
    elif args.dataset == 'PaviaU' or args.dataset == 'Pavia':
        red = 66
        green = 28
        blue = 0
    elif args.dataset == 'KSC':
        red = 28
        green = 14
        blue = 3
    elif args.dataset == 'Urban':
        red = 25
        green = 10
        blue = 0
    elif args.dataset == 'Washington':
        red = 54
        green = 34
        blue = 10
    elif args.dataset == 'DC':
        red = 54
        green = 34
        blue = 10
    # elif args.dataset == 'IndianP':
    #     red = 28
    #     green = 14
    #     blue = 3
    # elif args.dataset == 'MUUFL_HSI':
    #     red = 28
    #     green = 14
    #     blue = 3
    # elif args.dataset == 'Houston_HSI':
    #     red = 28
    #     green = 14
    #     blue = 3
    # elif args.dataset == 'Chikusei':
    #     red = 54
    #     green = 24
    #     blue = 3
    elif args.dataset == 'Augsburg':
        red = 48
        green = 28
        blue = 10
    elif args.dataset == 'Cave21':
        red = 30
        green = 15
        blue = 3


    lr = np.squeeze(test_lr.detach().cpu().numpy())
    lr = cv2.resize(lr, (out.shape[2], out.shape[3]), interpolation=cv2.INTER_NEAREST)
    slr = np.squeeze(lr).transpose(1, 2, 0).astype(np.float64)
    # io.savemat('./figs/11-21修改' +  '/' + '{}_{}_lr_subarea3_MIS.mat'.format(args.dataset, args.arch),{'Out': slr})

    lr_red = lr[red, :, :][:, :, np.newaxis]
    lr_green = lr[green, :, :][:, :, np.newaxis]
    lr_blue = lr[blue, :, :][:, :, np.newaxis]
    lr = np.concatenate((lr_blue, lr_green, lr_red), axis=2)
    lr = 255 * (lr - np.min(lr)) / (np.max(lr) - np.min(lr))
    lr = cv2.resize(lr, (out.shape[2], out.shape[3]), interpolation=cv2.INTER_NEAREST)
    # cv2.imwrite('./figs/11-21修改/{}_lr_subarea3.jpg'.format(args.dataset), lr)

    out = np.squeeze(out)# 移除尺寸为1的维度
    sout = np.squeeze(out).transpose(1, 2, 0).astype(np.float64)
    # io.savemat('./figs/11-21修改'+'/{}_{}_out_subarea3_MIS.mat'.format(args.dataset, args.arch),{'Out':sout})
    out_red = out[red, :, :][:, :, np.newaxis]
    out_green = out[green, :, :][:, :, np.newaxis]
    out_blue = out[blue, :, :][:, :, np.newaxis]
    out = np.concatenate((out_blue, out_green, out_red), axis=2)
    out = 255 * (out - np.min(out)) / (np.max(out) - np.min(out))
    # cv2.imwrite('./figs/11-21修改/{}_{}_out_subarea3_MIS.jpg'.format(args.dataset, args.arch), out)  #

    ref = np.squeeze(ref)
    sref = np.squeeze(ref).transpose(1, 2, 0).astype(np.float64)
    # io.savemat('./figs/11-21修改'+'/{}_{}_ref_subarea3_MIS.mat'.format(args.dataset, args.arch),
    #            {'Out': sref})
    ref_red = ref[red, :, :][:, :, np.newaxis]
    ref_green = ref[green, :, :][:, :, np.newaxis]
    ref_blue = ref[blue, :, :][:, :, np.newaxis]
    ref = np.concatenate((ref_blue, ref_green, ref_red), axis=2)
    ref = 255 * (ref - np.min(ref)) / (np.max(ref) - np.min(ref))
    # cv2.imwrite('./figs/11-21修改/{}_ref_subarea3_MIS.jpg'.format(args.dataset), ref)

    lr_dif = np.uint8(1.5 * np.abs((lr - ref)))  # lr
    lr_dif = cv2.cvtColor(lr_dif, cv2.COLOR_BGR2GRAY)
    lr_dif = cv2.applyColorMap(lr_dif, cv2.COLORMAP_JET)
    # cv2.imwrite('./figs/11-21修改/{}_lr_dif_subarea3.jpg'.format(args.dataset), lr_dif)

    out_dif = np.uint8(1.5 * np.abs((out - ref)))
    out_dif = cv2.cvtColor(out_dif, cv2.COLOR_BGR2GRAY)
    out_dif = cv2.applyColorMap(out_dif, cv2.COLORMAP_JET)
    # cv2.imwrite('./figs/11-21修改/{}_{}_out_dif_subarea3_MIS.jpg'.format(args.dataset, args.arch), out_dif)


if __name__ == '__main__':
    main()
