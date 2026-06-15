import os
import numpy as np
import torch
import torch.nn as nn
# from MCT_Net_main import args_parser
#
# args = args_parser.args_parser()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Moco(nn.Module):
    def __init__(self,base_encoder,image_lr,image_hr,dim=128,K=32*128,m=0.999,T=0.07,):
        super(Moco,self).__init__()

        self.K = K
        self.m = m
        self.T = T
        x_query = image_lr
        x_key = image_hr

        # create the encoders
        # num_classes is the output fc dimension
        # 构建查询编码器和键编码器。
        # self.encoder_q = base_encoder_q()  # 六层卷积
        self.encoder_q = base_encoder(x_query, x_key)#args.arch,  # MCT
                     # args.scale_ratio,
                     # args.n_select_bands,
                     # args.n_bands)  #  MCT_transformer
        self.encoder_k = base_encoder(x_query, x_key)#(args.arch,
                     # args.scale_ratio,
                     # args.n_select_bands,
                     # args.n_bands)
        # 在初始化时，将查询编码器的参数复制给键编码器，并将键编码器的参数设置为不需要梯度更新。
        # print(self.encoder_q)  # tuple类型：元祖
        # for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
        #     param_k.data.copy_(param_q.data)  # initialize
        #     param_k.requires_grad = False  # not update by gradient

        # create the queue 创建一个队列和一个指针，用于存储负样本和管理队列
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):  # 键编码器
        """
        Momentum update of the key encoder
        # 这是一个无梯度操作函数，用于更新键编码器的参数。它通过使用动量更新的方式将查询编码器的参数传递给键编码器。
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):  # 这是一个无梯度操作函数，用于更新队列。它将新的键添加到队列中，并移除队列中最早的键。
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    # 这两个函数用于在分布式训练环境下对输入进行批量洗牌和还原洗牌操作
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def normalize_tensor(tensor):
        # 计算张量的均值和标准差
        mean = torch.mean(tensor)
        std = torch.std(tensor)
        # 归一化张量
        normalized_tensor = (tensor - mean) / std
        return normalized_tensor

    def forward(self,im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if self.training:
            # compute query features
            # 索引0
            # q = self.encoder_q(im_q, im_k)[self.encoder_q(im_q, im_k) != 0]
            q,_ = self.encoder_q  # 元祖无法调用

            num_tensors = sum(isinstance(item, torch.Tensor) for item in q)

            # print("Number of tensors:", num_tensors)   # Number of tensors: 6
            # for tensor in q:
            #     print('shape',tensor.shape)

            '''Number of tensors: 6
shape torch.Size([1, 103, 128, 128])
shape torch.Size([1, 103, 128, 128])
shape torch.Size([1, 103, 128, 128])
shape torch.Size([1, 103, 127, 128])
shape torch.Size([1, 103, 128, 127])
shape torch.Size([1, 102, 128, 128])'''
            # q = torch.tensor(q,requires_grad=True)
            q.clone().detach().requires_grad_(True)
            q = nn.functional.normalize(q, dim=1)

            # compute key features
            # with torch.no_grad():  # no gradient to keys
            #     self._momentum_update_key_encoder()  # update the key encoder

                # _, k = self.encoder_k(im_q ,im_k)  # keys: NxC
            k,_ = self.encoder_k
            #
            # for tensor in k:
            #     print('shape k ',tensor.shape)
            k = torch.tensor(k,requires_grad=True)
            k = nn.functional.normalize(k, dim=1).detach()


            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            # 计算正样本的相似度 l_pos 和负样本的相似度 l_neg
            l_pos = torch.einsum('nijk,nijk->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('aecd,aecd->ad', [q, k] )# self.queue.clone().detach().unsqueeze(0)])

            # logits: Nx(1+K) 将正样本和负样本的相似度拼接成 logits

            # print("l_pos, l_neg",l_pos.shape, l_neg.shape)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T  # 对 logits 进行 softmax 温度调整

            # labels: positive key indicators
            # 创建标签 labels，其中正样本对应的标签为 0
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            # self._dequeue_and_enqueue(k)

            # return embedding, logits, labels
            return logits, labels
        # else:
        #     embedding, _ = self.encoder_q(im_q)

            # return embedding




# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output