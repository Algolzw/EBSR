import numpy as np
from datasets.synthetic_burst_train_set import SyntheticBurst
from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB
from datasets.burstsr_dataset import pack_raw_image, flatten_raw_image_batch


def main():
    train_zurich_raw2rgb = ZurichRAW2RGB(root='/data/dataset/ntire21/burstsr/synthetic', split='train')
    train_data = SyntheticBurst(train_zurich_raw2rgb, burst_size=14, crop_sz=384)
    means = []
    stds = []

    for data in train_data:
        burst, frame_gt, flow_vectors, meta_info = data
        burst = flatten_raw_image_batch(burst[None, ...]).squeeze().numpy()
        m = burst.mean(axis=(1, 2)).mean()
        s = burst.std(axis=(1, 2)).mean()
        means.append(m)
        stds.append(s)

        print(np.mean(means), np.mean(stds))
    # print(np.mean(datas), np.std(datas))


if __name__ == '__main__':
    # if not args.cpu: torch.cuda.set_device(0)
    main()
