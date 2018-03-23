class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    def update(self, val, n=1):
        self.val=val
        self.sum+=val*n
        self.count=n
        self.avg=self.sum/self.count

def accuracy(y_pred, gt, topk=(1,)):
    maxk=max(topk)
    batch_size=gt.size(0)

    _, pred=y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(gt.view(1, -1).expand_as(pred))

    res=[]
    for k in topk:
        correct_k=correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

