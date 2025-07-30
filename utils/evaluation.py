import torch
import numpy as np


class Evaluator(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sensing_margin = 0.01
        self.performance = self.nominal
        self.args = args

    def nominal(self, prediction, label):
        act, idx = torch.max(prediction, dim=1)
        corrects = (label.view(-1) == idx)
        return corrects.float().sum().item() / label.numel()

    def maa(self, prediction, label):
        act, idx = torch.topk(prediction, k=2, dim=1)
        corrects = (act[:, 0] >= self.sensing_margin) & (
            act[:, 1] <= 0) & (label.view(-1) == idx[:, 0])
        return corrects.float().sum().item() / label.numel()

    def norminal_snn(self, output, label):
        spk, _ = output
        act, idx = spk.sum(2).max(dim=1)
        corrects = (label.view(-1) == idx)
        return corrects.float().sum().item() / label.numel()

    def forward(self, nn, x, label):
        if self.args.metric == 'acc':
            self.performance = self.nominal
        elif self.args.metric == 'maa':
            self.performance = self.maa
        elif self.args.metric == 'acc_snn':
            self.performance = self.norminal_snn

        prediction = nn(x)
        F = prediction.shape[0]
        N = prediction.shape[1]
        acc = []
        for f in range(F):
            for n in range(N):
                acc.append(self.performance(prediction[f, n, :, :], label))
        accuracy = np.mean(acc)
        std = np.std(acc)
        return {'acc': accuracy, 'std': std, 'power': nn.Power, 'area': nn.Area}
        # return {'acc': accuracy, 'std': std}
