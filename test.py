
import cv2
import torch
import numpy as np
import os
from tqdm import tqdm
import random
import utility
from option import args

from datasets.synthetic_burst_val_set import SyntheticBurstVal
from datasets.burstsr_dataset import flatten_raw_image_batch
import model

import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import time


checkpoint = utility.checkpoint(args)

def sample_images(burst_size=14):
    _burst_size = 14

    ids = random.sample(range(1, _burst_size), k=burst_size - 1)
    ids = [0, ] + ids
    return ids


def ttaup(burst):
    burst0 = burst.clone()
    burst0 = flatten_raw_image_batch(burst0.unsqueeze(0)).cuda()

    burst3 = burst0.clone().permute(0, 1, 2, 4, 3).cuda()

    ids = sample_images(burst.shape[0])
    burst4 = burst0[:, ids].clone()

    return burst0, burst3, burst4


def ttadown(bursts):
    burst0 = bursts[0]

    burst3 = bursts[1].permute(0, 1, 3, 2)
    burst4 = bursts[2]

    out = (burst0 + burst3 + burst4) / 3
    return out


def main():
    mp.spawn(main_worker, nprocs=1, args=(1, args))


def main_worker(local_rank, nprocs, args):

    cudnn.benchmark = True
    args.local_rank = local_rank
    utility.setup(local_rank, nprocs)
    torch.cuda.set_device(local_rank)


    dataset = SyntheticBurstVal(args.root)
    out_dir = 'val'

    _model = model.Model(args, checkpoint)

    os.makedirs(out_dir, exist_ok=True)

    tt = []
    for idx in tqdm(range(len(dataset))):
        burst, burst_name = dataset[idx]
        bursts = ttaup(burst)

        srs = []
        with torch.no_grad():
            for x in bursts:
                tic = time.time()
                sr = _model(x, 0)
                toc = time.time()
                tt.append(toc-tic)
                srs.append(sr)

        sr = ttadown(srs)
        # Normalize to 0  2^14 range and convert to numpy array
        net_pred_np = (sr.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(np.uint16)
        cv2.imwrite('{}/{}.png'.format(out_dir, burst_name), net_pred_np)

    print('avg time: {:.4f}'.format(np.mean(tt)))
    utility.cleanup()


if __name__ == '__main__':
    main()
