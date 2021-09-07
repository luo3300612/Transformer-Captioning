import requests
import torch.nn as nn
import torch
import torch.nn.functional as F


def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    if 'drive.google.com' not in url:
        print('Downloading %s; may take a few minutes' % url)
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(path, "wb") as file:
            file.write(r.content)
        return
    print('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=None):
        super(LabelSmoothing, self).__init__()
        self.true_dist = None
        self.smoothing = smoothing
        self.confidence = 1.0 - self.smoothing
        self.ignore_index = ignore_index
        # self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.criterion = nn.KLDivLoss(reduction='none')

    def forward(self, logit, target_seq):
        logP = F.log_softmax(logit.view(-1, logit.shape[-1]), dim=-1)
        target_seq = target_seq.view(-1)
        mask = target_seq != self.ignore_index

        assign_seq = target_seq

        size = logP.size(1)
        true_dist = logP.clone()
        true_dist.fill_(self.smoothing / (size - 1))
        true_dist.scatter_(1, assign_seq.data.unsqueeze(1), self.confidence)
        loss = self.criterion(logP, true_dist).sum(1)
        loss = torch.masked_select(loss, mask).mean()
        return loss#, {'LabelSmoothing Loss': loss.item()}
