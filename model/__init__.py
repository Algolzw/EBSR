import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        self.args = args
        if args.local_rank == 0:
            print('Making model...')

        self.scale = args.scale
        self.idx_scale = 0
        self.input_large = (args.model == 'VDSR')
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda:%d' % args.local_rank)
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        # print(self.model.device)
        self.model = nn.parallel.DistributedDataParallel(self.model,
            device_ids=[args.local_rank],
            find_unused_parameters=True
            )
        if args.model == 'ENSEMBLE':
            self.model.module.model1.load_state_dict(
                torch.load(
                    '../train_log/ddp/real_models/wdcn_lrsc/ddpbest_epoch.pth',
                    {'cuda:0' : 'cuda:%d' % self.args.local_rank}).module.state_dict())
            self.model.module.model2.load_state_dict(
                torch.load(
                    '../train_log/ddp/real_models/lrsc_nfsdcn_finetune/ddpbest_epoch.pth',
                    {'cuda:0' : 'cuda:%d' % self.args.local_rank}).module.state_dict())
            self.model.module.model3.load_state_dict(
                torch.load(
                    '../train_log/ddp/real_models/lrscw_nfdcn/ddpbest_epoch.pth',
                    {'cuda:0' : 'cuda:%d' % self.args.local_rank}).module.state_dict())
        # self.model = DDP(self.model)
        if args.precision == 'half':
            self.model.half()

        self.load(
            ckp.get_path('model'),
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.model, file=ckp.log_file)

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

        if self.training:
            if self.n_GPUs > 1:
                return self.model(x)
        else:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function)
            else:
                # return self.model(x)
                return forward_function(x)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )
        if self.n_GPUs > 1:
            model = self.model.module
        else:
            model = self.model

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:
            if pre_train == 'download':
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=dir_model,
                    **kwargs
                )
            elif pre_train:
                print('Load the model from {}'.format(pre_train))
                map_location = {'cuda:%d' % 0: 'cuda:%d' % self.args.local_rank}
                load_from = torch.load(pre_train, map_location=map_location)
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        if load_from:
            # load_from = load_from.model
            if self.args.load_head:
                self.model.module.conv_first.load_state_dict(load_from.module.conv_first.state_dict())
                self.model.module.feature_extraction.load_state_dict(load_from.module.feature_extraction.state_dict())
                self.model.module.fea_L2_conv1.load_state_dict(load_from.module.fea_L2_conv1.state_dict())
                # self.model.module.fea_L2_conv2.load_state_dict(load_from.module.fea_L2_conv2.state_dict())
                self.model.module.fea_L3_conv1.load_state_dict(load_from.module.fea_L3_conv1.state_dict())
                # self.model.module.fea_L3_conv2.load_state_dict(load_from.module.fea_L3_conv2.state_dict())
                self.model.module.toplayer.load_state_dict(load_from.module.toplayer.state_dict())
                self.model.module.smooth1.load_state_dict(load_from.module.smooth1.state_dict())
                self.model.module.smooth2.load_state_dict(load_from.module.smooth2.state_dict())
                self.model.module.latlayer1.load_state_dict(load_from.module.latlayer1.state_dict())
                self.model.module.latlayer2.load_state_dict(load_from.module.latlayer2.state_dict())
                self.model.module.pcd_align.load_state_dict(load_from.module.pcd_align.state_dict())
                self.model.module.fusion.load_state_dict(load_from.module.fusion.state_dict())

            if self.args.load_sr:
                self.model.module.recon_trunk.load_state_dict(load_from.module.recon_trunk.state_dict())
                self.model.module.skipup1.load_state_dict(load_from.module.skipup1.state_dict())
                self.model.module.skipup2.load_state_dict(load_from.module.skipup2.state_dict())
                # self.model.module.recon_trunk_large.load_state_dict(load_from.module.recon_trunk_large.state_dict())
                self.model.module.upconv1.load_state_dict(load_from.module.upconv1.state_dict())
                self.model.module.upconv2.load_state_dict(load_from.module.upconv2.state_dict())
                self.model.module.HRconv.load_state_dict(load_from.module.HRconv.state_dict())
                self.model.module.conv_last.load_state_dict(load_from.module.conv_last.state_dict())


            if not self.args.load_head and not self.args.load_sr:
                self.model.load_state_dict(load_from.state_dict())


            if self.args.finetune_head:
                for param in self.model.module.parameters():
                    param.requires_grad = False
                for param in self.model.module.pcd_align.parameters():
                    param.requires_grad = True
                for param in self.model.module.fusion.parameters():
                    param.requires_grad = True

            if self.args.finetune_large:
                for param in self.model.module.parameters():
                    param.requires_grad = False
                for param in self.model.module.fusion.parameters():
                    param.requires_grad = True
                for param in self.model.module.recon_trunk[-1].parameters():
                    param.requires_grad = True
                for param in self.model.module.recon_trunk_large.parameters():
                    param.requires_grad = True
                for param in self.model.module.skipup1.parameters():
                    param.requires_grad = True
                for param in self.model.module.skipup2.parameters():
                    param.requires_grad = True
                for param in self.model.module.upconv1.parameters():
                    param.requires_grad = True
                for param in self.model.module.upconv2.parameters():
                    param.requires_grad = True
                for param in self.model.module.HRconv.parameters():
                    param.requires_grad = True
                for param in self.model.module.conv_last.parameters():
                    param.requires_grad = True
                if self.args.use_tree:
                    for param in self.model.module.trees1.parameters():
                        param.requires_grad = True
                    for param in self.model.module.trees2.parameters():
                        param.requires_grad = True

            if self.args.finetune_large_skip:
                for param in self.model.module.parameters():
                    param.requires_grad = False
                # for param in self.model.module.recon_trunk[-2].parameters():
                    # param.requires_grad = True
                for param in self.model.module.recon_trunk[-1].parameters():
                    param.requires_grad = True
                # for param in self.model.module.recon_trunk_large[-2].parameters():
                    # param.requires_grad = True
                for param in self.model.module.recon_trunk_large[-1].parameters():
                    param.requires_grad = True
                for param in self.model.module.recon_trunk_large_skip.parameters():
                    param.requires_grad = True
                for param in self.model.module.upconv1.parameters():
                    param.requires_grad = True
                for param in self.model.module.upconv2.parameters():
                    param.requires_grad = True
                for param in self.model.module.HRconv.parameters():
                    param.requires_grad = True
                for param in self.model.module.conv_last.parameters():
                    param.requires_grad = True

            if self.args.finetune_pcd:
                for param in self.model.module.parameters():
                    param.requires_grad = False
                for param in self.model.module.pcd_align.parameters():
                    param.requires_grad = True
                for param in self.model.module.tsa_fusion.parameters():
                    param.requires_grad = True
                for param in self.model.module.upconv1.parameters():
                    param.requires_grad = True
                for param in self.model.module.upconv2.parameters():
                    param.requires_grad = True
                for param in self.model.module.HRconv.parameters():
                    param.requires_grad = True
                for param in self.model.module.conv_last.parameters():
                    param.requires_grad = True

            del load_from

    def forward_chop(self, *args, shave=10, min_size=160000):
        scale = 1 if self.input_large else self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        # height, width
        h, w = args[0].size()[-2:]

        top = slice(0, h//2 + shave)
        bottom = slice(h - h//2 - shave, h)
        left = slice(0, w//2 + shave)
        right = slice(w - w//2 - shave, w)
        x_chops = [torch.cat([
            a[..., top, left],
            a[..., top, right],
            a[..., bottom, left],
            a[..., bottom, right]
        ]) for a in args]

        y_chops = []
        if h * w < 4 * min_size:
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
                y = P.data_parallel(self.model, *x, range(n_GPUs))
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*x_chops):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

        h *= scale
        w *= scale
        top = slice(0, h//2)
        bottom = slice(h - h//2, h)
        bottom_r = slice(h//2 - h, None)
        left = slice(0, w//2)
        right = slice(w - w//2, w)
        right_r = slice(w//2 - w, None)

        # batch size, number of color channels
        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if len(y) == 1: y = y[0]

        return y

    def forward_x8(self, *args, forward_function=None):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        list_x = []
        for a in args:
            x = [a]
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1: y = y[0]

        return y
