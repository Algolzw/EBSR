import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T

import utility
import model
import loss
from option import args
from trainer import Trainer
from datasets.synthetic_burst_train_set import SyntheticBurst
from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


checkpoint = utility.checkpoint(args)


def train_transform(train=False):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip())
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def main():
    mp.spawn(main_worker, nprocs=args.n_GPUs, args=(args.n_GPUs, args))


def main_worker(local_rank, nprocs, args):
    if checkpoint.ok:
        args.local_rank = local_rank
        init_seeds(local_rank+1)
        cudnn.benchmark = True
        utility.setup(local_rank, nprocs)
        torch.cuda.set_device(local_rank)

        train_trans = train_transform(train=True)

        batch_size = int(args.batch_size / nprocs)
        train_zurich_raw2rgb = ZurichRAW2RGB(root=args.root, split='train')
        train_data = SyntheticBurst(train_zurich_raw2rgb, burst_size=args.burst_size, crop_sz=args.patch_size)

        valid_zurich_raw2rgb = ZurichRAW2RGB(root=args.root, split='test')
        valid_data = SyntheticBurst(valid_zurich_raw2rgb, burst_size=args.burst_size, crop_sz=384)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data, shuffle=False)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=args.n_GPUs*2,
                                  pin_memory=True, drop_last=True, sampler=train_sampler)
        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size//2, num_workers=args.n_GPUs,
                                  pin_memory=True, drop_last=True, sampler=valid_sampler)

        _model = model.Model(args, checkpoint)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, train_loader, train_sampler, valid_loader, _model, _loss, checkpoint)
        while not t.terminate():
            t.train()

        del _model
        del _loss
        del train_loader
        del valid_loader

        utility.cleanup()

        checkpoint.done()


if __name__ == '__main__':
    main()
