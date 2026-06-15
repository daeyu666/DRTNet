from torch import nn
from utils import *
import cv2
import pdb
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam


def _as_tuple(outputs):
    if isinstance(outputs, (tuple, list)):
        return tuple(outputs)
    return (outputs,)


def _select_output(outputs, arch):
    outputs = _as_tuple(outputs)
    if len(outputs) >= 6:
        if arch in ('SSRSpat', 'SpatRNET'):
            return outputs[1]
        if arch in ('SSRSpec', 'SpecRNET'):
            return outputs[2]
    return outputs[0]


def validate(test_list, arch, model, epoch, n_epochs):
    test_ref, test_lr, test_hr = test_list
    model.eval()

    psnr = 0
    with torch.no_grad():
        # Set mini-batch dataset
        ref = to_var(test_ref).detach()
        lr = to_var(test_lr).detach()
        hr = to_var(test_hr).detach()
        out = _select_output(model(lr, hr), arch)

        ref = ref.detach().cpu().numpy()
        out = out.detach().cpu().numpy()

        rmse = calc_rmse(ref, out)
        psnr = calc_psnr(ref, out)
        ergas = calc_ergas(ref, out)
        sam = calc_sam(ref, out)

        with open('{}.txt'.format(arch), 'a') as f:
            f.write(str(epoch) + ',' + str(rmse) + ',' + str(psnr) + ',' + str(ergas) + ',' + str(sam) + ',' + '\n')

    return psnr
