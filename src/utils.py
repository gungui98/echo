import torch


def iou_metric_segmentation(pred, gt):
    """
    IoU metric for segmentation task
    """
    pred = pred.max(1)[1]
    gt = gt.max(1)[1]
    intersection = (pred == gt).sum()
    union = pred.size(0) + gt.size(0) - intersection
    return float(intersection) / float(union)


def iou_metric_segmentation_batch(pred, gt):
    """
    IoU metric for segmentation task
    """
    batch_size = pred.size(0)
    metric = 0
    for i in range(batch_size):
        metric += iou_metric_segmentation(pred[i], gt[i])
    return metric / batch_size


if __name__ == '__main__':
    pred = torch.randint(0, 10, (16, 10, 10))
    gt = torch.randint(0, 10, (16, 10, 10))
    print(iou_metric_segmentation(pred, gt))
    print(iou_metric_segmentation_batch(pred, gt))
