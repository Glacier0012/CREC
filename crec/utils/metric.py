# coding=utf-8


class ProgressMeter(object):
    def __init__(self, version,num_epochs, num_batches, meters, prefix=""):
        self.fmtstr = self._get_epoch_batch_fmtstr(version,num_epochs, num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, epoch, batch):
        entries = [self.prefix + self.fmtstr.format(epoch, batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_epoch_batch_fmtstr(self, version,num_epochs, num_batches):
        num_digits_epoch = len(str(num_epochs // 1))
        num_digits_batch = len(str(num_batches // 1))
        epoch_fmt = '{:' + str(num_digits_epoch) + 'd}'
        batch_fmt = '{:' + str(num_digits_batch) + 'd}'
        return '[' 'version: '+version+' '+ epoch_fmt + '/' + epoch_fmt.format(num_epochs) + ']' + '[' + batch_fmt + '/' + batch_fmt.format(
            num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        self.avg_reduce = 0.

    def update(self, val, n=1):
        self.val = val
        if n==-1:
            self.sum=val
            self.count=1
        else:
            self.sum += val * n
            self.count += n
        self.avg = self.sum / self.count

    def update_reduce(self, val):
        self.avg_reduce = val

    def __str__(self):
        fmtstr = '{name} {avg_reduce' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

