import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        #print(output)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        #print(pred[0])
        #print(target)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True).cpu().numpy()
            res[k] = (correct_k / batch_size)
        return res


# Quadratic Expansion
def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def intersection(bbox_pred, bbox_gt):
    max_xy = torch.min(bbox_pred[:, 2:], bbox_gt[:, 2:])
    min_xy = torch.max(bbox_pred[:, :2], bbox_gt[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]


def jaccard(bbox_pred, bbox_gt):
    inter = intersection(bbox_pred, bbox_gt)
    area_pred = (bbox_pred[:, 2] - bbox_pred[:, 0]) * (bbox_pred[:, 3] -
      bbox_pred[:, 1])
    area_gt = (bbox_gt[:, 2] - bbox_gt[:, 0]) * (bbox_gt[:, 3] -
      bbox_gt[:, 1])
    union = area_pred + area_gt - inter
    iou = torch.div(inter, union)
    return torch.sum(iou)

def get_total_norm(parameters, norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
    for p in parameters:
        try:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
            total_norm = total_norm ** (1. / norm_type)
        except:
            continue
    return total_norm

def retrieval_map_evaluation(test_states, test_targets, cate_num, collection_states=None, collection_targets=None, topk=(1,5), batch_size=32):
    collection_is_test = False
    if collection_states is None:
        collection_states = test_states
        collection_targets = test_targets
        collection_is_test = True
    num_test_batch = test_states.size(0) // batch_size + 1
    num_collect_batch = collection_states.size(0) // batch_size + 1
    MAPs = []
    for test_i in range(num_test_batch):
        test_i_distances = []
        for collect_i in range(num_collect_batch):
            test_batch = test_states[test_i*batch_size:(test_i+1)*batch_size,:] #[batch, feat_dim]
            collect_batch = collection_states[collect_i*batch_size:(collect_i+1)*batch_size,:] #[batch, feat_dim]
            tmp_distances = pairwise_distances(test_batch, collect_batch) # [batch, batch]
            if collection_is_test and test_i == collect_i:
                tmp_distances[torch.arange(tmp_distances.size(0)), torch.arange(tmp_distances.size(0))] = 1e+8
            test_i_distances.append(tmp_distances)
        test_i_distances = -torch.cat(test_i_distances, dim=1) #[batch, length]
        _ , pred_indics = test_i_distances.sort(dim=1)
        #test_indics = [i for i in range(test_i*batch_size, (test_i+1)*batch_size)]
        #print(test_indics, test_targets)

        test_i_target = test_targets[test_i*batch_size:(test_i+1)*batch_size]

        # pred_i_target = collection_targets[pred_indics]
        for j in range(len(test_i_target)):
            c = test_i_target[j]
            res = (collection_targets[pred_indics[j]] == c).to(torch.float)
            k, rightk, precision = 0, 0, []
            while rightk < cate_num[c]:
                r = res[k].item()
                if r:
                    precision.append((res[:k + 1]).mean().item())
                    rightk += 1
                k += 1
            MAPs.append(sum(precision) / len(precision))
    MAP = sum(MAPs) / len(MAPs)
    return MAP

def retrieval_accuracy_evaluation(test_states, test_targets, collection_states=None, collection_targets=None, topk=(1,5), batch_size=32):
    collection_is_test = False
    if collection_states is None:
        collection_states = test_states
        collection_targets = test_targets
        collection_is_test = True
    num_test_batch = test_states.size(0) // batch_size + 1
    num_collect_batch = collection_states.size(0) // batch_size + 1
    correct = {k:0 for k in  topk}
    for test_i in range(num_test_batch):
        test_i_distances = []
        for collect_i in range(num_collect_batch):
            test_batch = test_states[test_i*batch_size:(test_i+1)*batch_size,:] #[batch, feat_dim]
            collect_batch = collection_states[collect_i*batch_size:(collect_i+1)*batch_size,:] #[batch, feat_dim]
            tmp_distances = pairwise_distances(test_batch, collect_batch) # [batch, batch]
            if collection_is_test and test_i == collect_i:
                tmp_distances[torch.arange(tmp_distances.size(0)), torch.arange(tmp_distances.size(0))] = 1e+8
            test_i_distances.append(tmp_distances)
        test_i_distances = -torch.cat(test_i_distances, dim=1) #[batch, length]
        _ , pred_indics = test_i_distances.topk(max(topk), dim=1)
        #test_indics = [i for i in range(test_i*batch_size, (test_i+1)*batch_size)]
        #print(test_indics, test_targets)

        test_i_target = test_targets[test_i*batch_size:(test_i+1)*batch_size]
        # pred_i_target = collection_targets[pred_indics]
        for j in range(len(test_i_target)):
            for k in topk:
                if test_i_target[j] in collection_targets[pred_indics[j][:k]]:
                    correct[k] = correct[k] + 1

    return {'retrieval_{}'.format(k):(correct[k] / test_states.size(0)) for k in topk}
