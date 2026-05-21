from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import yaml
from scipy import interpolate
from PIL import Image


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


class MetricLogger(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressLogger(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def metric_AUROC(target, output, nb_classes=1):
    """
    Compute per-class AUROC.
    Silently skips classes where all labels are the same (no AUC defined).

    target, output : FloatTensors on any device, shape (N, nb_classes)
    Returns list of AUC values (length ≤ nb_classes).
    """
    outAUROC = []
    target = target.cpu().numpy()
    output = output.cpu().numpy()

    for i in range(nb_classes):
        col_t = target[:, i] if target.ndim == 2 else target.flatten()
        col_p = output[:, i] if output.ndim == 2 else output.flatten()
        if len(np.unique(col_t)) > 1:
            outAUROC.append(roc_auc_score(col_t, col_p))

    return outAUROC


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters    = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = (final_value
                + 0.5 * (base_value - final_value)
                * (1 + np.cos(np.pi * iters / len(iters))))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_image(input_arr, idx):
    def _norm(d):
        lo, hi = d.min(), d.max()
        return np.uint8((d - lo) * 255.0 / (hi - lo + 1e-8))
    im = Image.fromarray(_norm(input_arr))
    im.save(f"{idx}.jpeg")


def remap_pretrained_keys_swin(model, checkpoint_model):
    state_dict = model.state_dict()
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            rpbt_pre = checkpoint_model[key]
            rpbt_cur = state_dict.get(key)
            if rpbt_cur is None:
                continue
            L1, nH1 = rpbt_pre.size()
            L2, nH2 = rpbt_cur.size()
            if nH1 != nH2:
                continue
            if L1 != L2:
                src_size = int(L1 ** 0.5)
                dst_size = int(L2 ** 0.5)
                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = sum(q**i for i in range(src_size // 2))
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q
                dis, cur = [], 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)
                r_ids = [-x for x in reversed(dis)]
                x = y = r_ids + [0] + dis
                t  = dst_size // 2.0
                dx = dy = np.arange(-t, t + 0.1, 1.0)
                all_bias = []
                for h in range(nH1):
                    z = rpbt_pre[:, h].view(src_size, src_size).float().numpy()
                    fc = interpolate.interp2d(x, y, z, kind='cubic')
                    all_bias.append(
                        torch.tensor(fc(dx, dy)).view(-1, 1).to(rpbt_pre.device))
                checkpoint_model[key] = torch.cat(all_bias, dim=-1)

    for k in [k for k in checkpoint_model if "relative_position_index" in k
              or "relative_coords_table" in k or "attn_mask" in k]:
        del checkpoint_model[k]

    return checkpoint_model
