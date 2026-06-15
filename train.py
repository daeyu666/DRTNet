import torch
from torch import nn
from utils import to_var, batch_ids2words
import random
import torch.nn.functional as F
import cv2


def spatial_edge(x):
    edge1 = x[:, :, 0:x.size(2) - 1, :] - x[:, :, 1:x.size(2), :]
    edge2 = x[:, :, :, 0:x.size(3) - 1] - x[:, :, :, 1:x.size(3)]

    return edge1, edge2


def spectral_edge(x):
    edge = x[:, 0:x.size(1) - 1, :, :] - x[:, 1:x.size(1), :, :]

    return edge


def mrae_loss(img_fus, img_tgt):
    absolute_error = torch.abs(img_fus - img_tgt)
    relative_error = absolute_error / (torch.abs(img_tgt) + 1e-8)
    mrae = torch.mean(relative_error)
    return mrae


def train(train_list,
          image_size,
          scale_ratio,
          n_bands,
          arch,
          model,
          optimizer,
          criterion,
          L1,
          epoch,
          n_epochs):
    train_ref, train_lr, train_hr = train_list

    h, w = train_ref.size(2), train_ref.size(3)
    h_str = random.randint(0, h - image_size - 1)
    w_str = random.randint(0, w - image_size - 1)

    train_ref = train_ref[:, :, h_str:h_str + image_size, w_str:w_str + image_size]
    train_lr = F.interpolate(train_ref, scale_factor=1 / (scale_ratio * 1.0))
    train_hr = train_hr[:, :, h_str:h_str + image_size, w_str:w_str + image_size]

    image_lr = to_var(train_lr).detach()
    image_hr = to_var(train_hr).detach()
    image_ref = to_var(train_ref).detach()

    optimizer.zero_grad()
    out, _ = model(image_lr, image_hr)

    if 'RNET' in arch:
        loss_fus = criterion(out, image_ref)
        loss_spat = criterion(out_spat, image_ref)
        loss_spec = criterion(out_spec, image_ref)
        loss_spec_edge = criterion(edge_spec, ref_edge_spec)
        loss_spat_edge = 0.5 * criterion(edge_spat1, ref_edge_spat1) + 0.5 * criterion(edge_spat2, ref_edge_spat2)
        if arch == 'SpatRNET':
            loss = loss_spat + loss_spat_edge
        elif arch == 'SpecRNET':
            loss = loss_spec + loss_spec_edge
        elif arch == 'SSRNET':
            loss = loss_fus
        elif arch == 'MCT':
            loss = loss_fus
        elif arch == 'MCT_RCB':
            loss = loss_fus
        elif arch == 'no_contrast':
            loss = criterion(out, image_ref)
            L1 = L1(out, image_ref)
            loss.backward()
    else:
        loss = criterion(out, image_ref)
        loss.backward()

    optimizer.step()

    print('Epoch [%d/%d], Loss: %.4f'
          % (epoch,
             n_epochs,
             loss,
             )
          )


'''train_contrast'''


def train_contrast(train_list,
                   image_size,
                   scale_ratio,
                   n_bands,
                   arch,
                   model,
                   E,
                   optimizer,
                   criterion,
                   loss_contrast,
                   epoch,
                   n_epochs):
    train_ref, train_lr, train_hr = train_list

    h, w = train_ref.size(2), train_ref.size(3)
    h_str = random.randint(0, h - image_size - 1)
    w_str = random.randint(0, w - image_size - 1)

    train_ref = train_ref[:, :, h_str:h_str + image_size, w_str:w_str + image_size]
    train_lr = F.interpolate(train_ref, scale_factor=1 / (scale_ratio * 1.0))
    train_hr = train_hr[:, :, h_str:h_str + image_size, w_str:w_str + image_size]

    image_lr = to_var(train_lr).detach()
    image_hr = to_var(train_hr).detach()
    image_ref = to_var(train_ref).detach()

    optimizer.zero_grad()

    out, x_spec = model(image_lr, image_hr)
    E.train()

    loss = criterion(out, image_ref)
    loss_spec = criterion(x_spec, image_ref)
    contrast_loss = E(out, image_ref, image_hr)
    loss_total = loss + loss_spec + contrast_loss

    loss_total.backward()
    optimizer.step()

    print('Epoch [%d/%d], Loss: %.4f, Spectral Loss: %.4f, CESR Loss: %.4f, Total: %.4f'
          % (epoch,
             n_epochs,
             loss,
             loss_spec,
             contrast_loss,
             loss_total,
             )
          )
