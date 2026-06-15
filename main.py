import random
import torch.backends.cudnn as cudnn
import torch.optim
from models.MIMO import MIMO
from models.SSRNET import SSRNET
from models.SingleCNN import SpatCNN, SpecCNN
from models.SSFCNN import SSFCNN, ConSSFCNN
from models.MCT_rectangle import MCT_rectangle
from utils import *
from data_loader import build_datasets
from validate import validate
from train import train,train_contrast
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
    elif args.dataset == 'DC':
        args.n_bands = 191
    elif args.dataset == 'MUUFL_HSI':
        args.n_bands = 64
    elif args.dataset == 'salinas_corrected':
        args.n_bands = 204
    elif args.dataset == 'Houston_HSI':
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

    elif args.arch == 'DRTnet':  # 11.21
        model = MCT_rectangle(args.arch,
                       args.scale_ratio,
                       args.n_select_bands,
                       args.n_bands,
                       args.dataset,
                       ).cuda()

    elif args.arch == 'MIMO':
        model =MIMO(args.n_select_bands,args.n_bands).cuda()

    ####################################################0424
    import random
    image_size = args.image_size
    scale_ratio = args.scale_ratio
    train_ref, train_lr, train_hr = train_list

    h, w = train_ref.size(2), train_ref.size(3)
    h_str = random.randint(0, h - image_size - 1)
    w_str = random.randint(0, w - image_size - 1)

    train_ref = train_ref[:, :, h_str:h_str + image_size, w_str:w_str + image_size]
    train_lr = F.interpolate(train_ref, scale_factor=1 / (scale_ratio * 1.0))  #  scale_factor 1
    train_hr = train_hr[:, :, h_str:h_str + image_size, w_str:w_str + image_size]

    # model.train()

    # Set mini-batch dataset
    image_lr = to_var(train_lr).detach()
    image_hr = to_var(train_hr).detach()
    image_ref = to_var(train_ref).detach()

    #################################################################################

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
    # MRAE = MRAELoss()
    L1 = nn.L1Loss().cuda()
    contrast_loss = torch.nn.CrossEntropyLoss().cuda()

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
        # One epoch's traininginceptionv3
        print('Train_Epoch_{}: '.format(epoch))
        # if epoch < args.epochs_encoder:
        # MOCO  # constrat learning
        E = Moco(base_encoder=model, image_lr=image_lr, image_hr=image_hr).cuda()
        train_contrast(train_list,
                       args.image_size,
                       args.scale_ratio,
                       args.n_bands,
                       args.arch,
                       model,
                       E,
                       optimizer,
                       criterion,  # MSELoss
                       contrast_loss,
                       epoch,
                       args.n_epochs)


        # One epoch's validation
        print('Val_Epoch_{}: '.format(epoch))
        recent_psnr = validate(test_list,
                               args.arch,
                               model,
                               epoch,
                               args.n_epochs)
        print('psnr: ', recent_psnr)

        # # save model
        is_best = recent_psnr > best_psnr
        best_psnr = max(recent_psnr, best_psnr)
        # if epoch > 9000 and epoch % 50 == 0:
        #     model_path_ = model_path.split('.pkl')[0] + 'ep' + str(epoch) + '.pkl'
        #     print(model_path_)
        #     torch.save(model.state_dict(), model_path_)
        if is_best:
            best_epoch = epoch
            if best_psnr > 0:
                torch.save(model.state_dict(), model_path)
                print('Saved!')
                print('')
        print('best psnr:', best_psnr, 'at epoch:', best_epoch)

    print('best_psnr: ', best_psnr)


# criterion(out, image_ref)
# def mrae_loss( img_fus, img_tgt):
#     # error = torch.abs(img_fus - img_tgt) / img_tgt
#     absolute_error = torch.abs(img_fus - img_tgt)
#     relative_error = absolute_error / (torch.abs(img_tgt)+ 1e-8)
#     mrae = torch.mean(relative_error)
#     return mrae

# class MRAELoss:
#     def __init__(self):
#         pass
#
#     def __call__(self, img_fus, img_tgt):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # 将输入数据移动到设备上
#         img_fus = img_fus.to(device)
#         img_tgt = img_tgt.to(device)
#
#         # absolute_error = torch.abs(img_fus - img_tgt)
#         # relative_error = absolute_error / (torch.abs(img_tgt) + 1e-8)
#         # mrae = torch.mean(relative_error)
#
#         error = torch.abs(img_fus - img_tgt) / img_tgt
#         mrae = torch.mean(error.reshape(-1))
#         return mrae

class MRAELoss(nn.Module):
    def __init__(self):
        super(MRAELoss, self).__init__()

    def forward(self, img_fus, img_tgt):
        assert img_fus.shape == img_tgt.shape
        error = torch.abs(img_fus - img_tgt) / img_tgt
        mrae = torch.mean(error.reshape(-1))
        return mrae


class Encoder_q(nn.Module):
    def __init__(self):
        super(Encoder_q, self).__init__()
        # 六层卷积操作
        self.E = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        # print('xinput',x.shape)  # ([1, 5, 128, 128])
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out


class Encoder_k(nn.Module):
    def __init__(self):
        super(Encoder_k, self).__init__()
        # 六层卷积操作
        self.E = nn.Sequential(
            nn.Conv2d(args.n_bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        # print('xinput',x.shape)  # ([1, 5, 128, 128])
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out

eps = 1e-7


class NCECriterion(nn.Module):

    def __init__(self, nLem):
        super(NCECriterion, self).__init__()
        self.nLem = nLem

    def forward(self, x, targets):
        # x shape: [batchSize, K+1]
        # targets shape: [batchSize]
        # K is the number of noise samples
        batchSize = x.size(0)
        K = x.size(1) - 1
        Pnt = 1 / float(self.nLem)  # P(origin=noise)
        Pns = 1 / float(self.nLem)  # P(noise=sample)

        # eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt)
        Pmt = x.select(1, 0)  # 1st column is the model output
        Pmt_div = Pmt.add(K * Pnt + eps)
        lnPmt = torch.div(Pmt, Pmt_div)

        # eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
        Pon_div = x.narrow(1, 1, K).add(K * Pns + eps)  # 2nd to last column are noise samples
        Pon = Pon_div.clone().fill_(K * Pns)
        lnPon = torch.div(Pon, Pon_div)

        # equation 6 in ref. A
        lnPmt.log_()
        lnPon.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.view(-1, 1).sum(0)

        loss = - (lnPmtsum + lnPonsum) / batchSize

        return loss


if __name__ == '__main__':
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    main()
