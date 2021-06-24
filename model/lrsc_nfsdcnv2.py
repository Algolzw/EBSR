''' network architecture for EDVR '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.arch_util as arch_util
from model.common import lanczos_shift

try:
    from model.non_local.non_local_cross_dot_product import NONLocalBlock2D as NonLocalCross
    from model.non_local.non_local_dot_product import NONLocalBlock2D as NonLocal
except ImportError:
    raise ImportError('Failed to import Non_Local module.')

try:
    from model.DCNv2.dcn_v2 import DCN_sep as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

def make_model(args, parent=False):
    return LRSC_NFSDCN(args)


class ShiftAlign(nn.Module):
    '''Shift Alignment
    '''
    def __init__(self, nf=64):
        super(ShiftAlign, self).__init__()
        self.location = nn.Sequential(nn.Conv2d(nf, nf, 3, 2, padding=1),
                                      nn.LeakyReLU(),

                                      nn.AdaptiveAvgPool2d(1),
                                      nn.Flatten(),
                                      nn.Linear(nf, 2))

    def forward(self, input, source):

        loc = self.location(input)

        return lanczos_shift(source, loc).contiguous()


class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8, wn=None):
        super(PCD_Align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for diff
        self.L3_offset_conv2 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        # self.L3_shift = ShiftAlign(nf)
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
                              # extra_offset_mask=True)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for diff
        self.L2_offset_conv2 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for offset
        self.L2_offset_conv3 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        # self.L2_shift = ShiftAlign(nf)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
                              # extra_offset_mask=True)
        self.L2_fea_conv = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for diff
        self.L1_offset_conv2 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for offset
        self.L1_offset_conv3 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        # self.L1_shift = ShiftAlign(nf)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
                              # extra_offset_mask=True)
        self.L1_fea_conv = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for diff
        self.cas_offset_conv2 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        # L3_nbr_fea = self.L3_shift(L3_offset, nbr_fea_l[2])
        L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset))
        # L2
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset*2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        # L2_nbr_fea = self.L2_shift(L2_offset, nbr_fea_l[1])
        L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        # L1_nbr_fea = self.L1_shift(L1_offset, nbr_fea_l[0])
        L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack(L1_fea, offset))

        return L1_fea


class NonLocal_Fusion(nn.Module):
    ''' Non Local fusion module
    '''
    def __init__(self, nf=64, nframes=5, center=2, wn=None):
        super(NonLocal_Fusion, self).__init__()
        self.center = center

        self.non_local_T = nn.ModuleList()
        self.non_local_F = nn.ModuleList()

        for i in range(nframes):
            self.non_local_T.append(NonLocalCross(nf, inter_channels=nf//2, sub_sample=True, bn_layer=False))
            self.non_local_F.append(NonLocal(nf, inter_channels=nf//2, sub_sample=True, bn_layer=False))

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = wn(nn.Conv2d(nframes * nf * 2, nf, 1, 1, bias=True))

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


class LRSC_NFSDCN(nn.Module):
    def __init__(self, args):
        super(LRSC_NFSDCN, self).__init__()

        nf = args.n_feats
        n_resblocks = args.n_resblocks
        nframes = args.burst_size
        front_RBs=5
        back_RBs=args.n_resgroups
        groups=8

        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.center = 0
        # self.center = nframes // 2 if center is None else center
        self.non_local = args.non_local # False
        self.channel_att = args.CA

        if self.channel_att == True:
            ResidualBlock_noBN_f = functools.partial(arch_util.WideActResBlock, nf=nf)
            ResidualBlock_noBN_SE = functools.partial(arch_util.LRSCResidualGroup, n_feat=nf, n_resblocks=n_resblocks, da=args.DA)
        else:
            ResidualBlock_noBN_f = functools.partial(arch_util.WideActResBlock, nf=nf)

        #### extract features (for each frame)
        self.conv_first = wn(nn.Conv2d(args.burst_channel, nf, 3, 1, 1, bias=True))
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        # self.feature_extraction = nn.Sequential(
        #     arch_util.make_layer_idx(ResidualBlock_noBN_f, front_RBs),
        #     wn(nn.Conv2d(nf*(front_RBs+1), nf, 1)))
        self.fea_L2_conv1 = wn(nn.Conv2d(nf, nf*2, 3, 2, 1, bias=True))
        self.fea_L3_conv1 = wn(nn.Conv2d(nf*2, nf*4, 3, 2, 1, bias=True))

        # Top layers
        self.toplayer = wn(nn.Conv2d(nf*4, nf, kernel_size=1, stride=1, padding=0))
        # Smooth layers
        self.smooth1 = wn(nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1))
        self.smooth2 = wn(nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1))
        # Lateral layers
        self.latlayer1 = wn(nn.Conv2d(nf*2, nf, kernel_size=1, stride=1, padding=0))
        self.latlayer2 = wn(nn.Conv2d(nf*1, nf, kernel_size=1, stride=1, padding=0))

        self.pcd_align = PCD_Align(nf=nf, groups=groups, wn=wn)
        if self.non_local:
            if args.local_rank == 0:
                print('use non_local')
            # self.non_local_fusion = NonLocal_Fusion(nf=nf, nframes=nframes, center=self.center)
            self.fusion = NonLocal_Fusion(nf=nf, nframes=nframes, center=self.center, wn=wn)
        else:
            # self.non_local_fusion = wn(nn.Conv2d(nframes * nf, nf, 1, 1, bias=True))
            self.fusion = wn(nn.Conv2d(nframes * nf, nf, 1, 1, bias=True))

        #### reconstruction
        if self.channel_att == False:
            self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
            # self.recon_trunk = nn.Sequential(
            #     arch_util.make_layer_idx(ResidualBlock_noBN_f, back_RBs),
            #     wn(nn.Conv2d(nf*(back_RBs+1), nf, 1)))
        else:
            self.recon_trunk = nn.Sequential(
                arch_util.make_layer_idx(ResidualBlock_noBN_SE, back_RBs),
                wn(nn.Conv2d(nf*(back_RBs+1), nf, 1)))
        #### upsampling
        self.upconv1 = wn(nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True))
        self.upconv2 = wn(nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True))
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = wn(nn.Conv2d(64, 64, 3, 1, 1, bias=True))
        self.conv_last = wn(nn.Conv2d(64, args.n_colors, 3, 1, 1, bias=True))

        #### skip #############
        self.skip_pixel_shuffle = nn.PixelShuffle(2)
        self.skipup1 = wn(nn.Conv2d(args.burst_channel, nf * 4, 3, 1, 1, bias=True))
        self.skipup2 = wn(nn.Conv2d(nf, args.n_colors * 4, 3, 1, 1, bias=True))

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def _upsample_add(self, x, y):
        # _,_,H,W = y.size()
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # print('x: ', x.shape)
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()

        #### skip module ########
        skip1 = self.lrelu2(self.skip_pixel_shuffle(self.skipup1(x_center)))
        skip2 = self.lrelu2(self.skip_pixel_shuffle(self.skipup2(skip1)))

        #### extract LR features
        # L1
        L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))

        L3_fea = self.lrelu(self.toplayer(L3_fea))
        L2_fea = self.smooth1(self._upsample_add(L3_fea, self.latlayer1(L2_fea)))
        L1_fea = self.smooth2(self._upsample_add(L2_fea, self.latlayer2(L1_fea)))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        #### pcd align
        # ref feature list
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W] --> [B, T, C, H, W]

        if not self.non_local:
            aligned_fea = aligned_fea.view(B, -1, H, W)

        fea = self.lrelu(self.fusion(aligned_fea))

        out = self.lrelu(self.recon_trunk(fea))
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = skip1 + out

        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)

        out = skip2 + out
        return out
