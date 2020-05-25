import torch
import numpy as np
from torchvision.datasets import Cityscapes

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Pixel Accuracy' : acc,
                'Frequency Weighted IoU': fwavacc,
                'Mean IoU': mean_iu,
                'Class IoU': cls_iu}


    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def calculate_loss(predictions, labels, loss_function):
    with torch.no_grad():
        average_loss = loss_function(predictions, labels)
    return average_loss.item()


def run_inference(data_loader, model, print_freq=50):
    model.netG.eval()
    scores = runningScore(model.num_classes)
    loss_function = model.criterion
    losses = []
    for i, (input_image, label) in enumerate(data_loader):
        input_image, label = input_image.to(model.device), label.to(model.device)
        prediction = get_prediction(input_image, model)
        losses.append(calculate_loss(prediction, label, loss_function))
        prediction_argmax = torch.argmax(prediction.cpu(), dim=1)
        prediction_argmax[label == 255] = 255
        scores.update(label.cpu().numpy(), prediction_argmax.numpy())
        assert label.shape != prediction_argmax, \
            f"shape of label and prediction are not the same (label/pred): {label.shape}/{prediction_argmax.shape}"
        if i > 0 and i % print_freq == 0:
            print(f"Finished {i}/{len(data_loader)}")
    print(f"Finished {len(data_loader)}/{len(data_loader)}")
    model.netG.train()
    return scores, np.mean(losses)


def get_prediction(input_image, model):
    with torch.no_grad():
        prediction = model.netG(input_image)
    return prediction

def evaluate(train_loader, val_loader, model):
    train_id_to_name = {cls[2]:cls[0] for cls in Cityscapes.classes if not cls[6]}

    train_scores, train_loss = run_inference(train_loader, model)
    train_stats = train_scores.get_scores()


    val_scores, val_loss = run_inference(val_loader, model)
    val_stats = val_scores.get_scores()

    model.store_losses((train_loss, val_loss))
    model.store_ious((train_stats['Mean IoU'], val_stats['Mean IoU']))
    print(f"Mean IoU: {train_stats['Mean IoU']}/{val_stats['Mean IoU']}")
    print(f"Frequency Weighted IoU: {train_stats['Frequency Weighted IoU']}/{val_stats['Frequency Weighted IoU']}")
    print(f"Pixel acc: {train_stats['Pixel Accuracy']}/{val_stats['Pixel Accuracy']}")
    print(f"Loss: {train_loss}/{val_loss}")
    for i in range(len(train_stats['Class IoU'])):
        print(f"IoU for {train_id_to_name[i]}: {train_stats['Class IoU'][i]}/{val_stats['Class IoU'][i]}")
    model.plot_loss()
    model.plot_iou()


