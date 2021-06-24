import torch
import torch.nn as nn
import torch.nn.functional as F
import model.arch_util as arch_util
from torch.cuda.amp import autocast
from model.lrsc_edvr import make_model as make_lrsc_edvr
from model.lrsc_nfsdcnv2 import make_model as make_lrsc_nfsdcnv2
from model.lrscw_nfdcn import make_model as make_lrscw_nfdcn

try:
    from model.non_local.non_local_cross_dot_product import NONLocalBlock2D as NonLocalCross
    from model.non_local.non_local_dot_product import NONLocalBlock2D as NonLocal
except ImportError:
    raise ImportError('Failed to import Non_Local module.')


def make_model(args, parent=False):
    return EnsembleModel(args)

class NonLocal_Fusion(nn.Module):
    ''' Non Local fusion module
    '''
    def __init__(self, nf=64, nframes=5, center=0):
        super(NonLocal_Fusion, self).__init__()
        self.center = center

        self.non_local_T = nn.ModuleList()
        self.non_local_F = nn.ModuleList()

        for i in range(nframes):
            self.non_local_T.append(NonLocalCross(nf, inter_channels=nf//4, sub_sample=True, bn_layer=False))
            self.non_local_F.append(NonLocal(nf, inter_channels=nf//4, sub_sample=True, bn_layer=False))

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf * 2, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        ref = aligned_fea[:, self.center, :, :, :].clone()

        cor_l = []
        non_l = []
        for i in range(N):
            nbr = aligned_fea[:, i, :, :, :]
            non_l.append(self.non_local_F[i](nbr))
            cor_l.append(self.non_local_T[i](nbr, ref))

        aligned_fea_T = torch.cat(cor_l, dim=1)
        aligned_fea_F = torch.cat(non_l, dim=1)
        aligned_fea = torch.cat([aligned_fea_T, aligned_fea_F], dim=1)

        #### fusion
        fea = self.fea_fusion(aligned_fea)

        return fea

class EnsembleModel(nn.Module):
    def __init__(self, args):
        super(EnsembleModel, self).__init__()

        args.n_feats = 128
        args.n_resblocks = 10
        self.model1 = make_lrsc_edvr(args)
        # self.model1.load_state_dict(torch.load('../train_log/ddp/real_models/wdcn_lrsc/ddpbest_epoch.pth').module.state_dict())
        if args.local_rank == 0:
            print('load model1 ...')
        for param in self.model1.parameters():
            param.requires_grad = False

        args.n_feats = 128
        args.n_resblocks = 20
        args.n_resgroups = 40
        args.non_local = True
        self.model2 = make_lrsc_nfsdcnv2(args)
        # self.model2.load_state_dict(torch.load('../train_log/ddp/real_models/lrsc_nfsdcn_finetune/ddpbest_epoch.pth').module.state_dict()) #, map_location = {'cuda:0': 'cpu'}
        if args.local_rank == 0:
            print('load model2 ...')
        for param in self.model2.parameters():
            param.requires_grad = False

        args.n_feats = 128
        args.n_resblocks = 8
        args.n_resgroups = 5
        args.non_local = True
        args.CA = True
        self.model3 = make_lrscw_nfdcn(args)
        # self.model3.load_state_dict(torch.load('../train_log/ddp/real_models/lrscw_nfdcn/ddpbest_epoch.pth').module.state_dict())
        if args.local_rank == 0:
            print('load model3 ...')
        for param in self.model3.parameters():
            param.requires_grad = False

        self.ensemble_head = nn.Sequential(
            nn.Conv2d(args.n_colors, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, True),
            arch_util.WideActResBlock(nf=64))

        self.conv_last = nn.Sequential(
            nn.Conv2d(3 * 64, 64, 1, 1, bias=True),
            nn.LeakyReLU(0.1, True),
            arch_util.WideActResBlock(nf=64),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, args.n_colors, 3, 1, 1, bias=True))

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x = torch.stack([x1, x2, x3], dim=1)
        B, N, C, H, W = x.size()

        x = self.ensemble_head(x.view(-1, C, H, W)).view(B, -1, H, W)
        x = self.conv_last(x) + x1

        return x

















