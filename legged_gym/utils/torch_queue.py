import torch

class TorchQueue:
    def __init__(self, maxlen, shape, dtype=torch.float32, device='cpu'):
        self.maxlen = maxlen
        self.data = torch.zeros((maxlen, *shape), dtype=dtype, device=device)
        self.size = 0
        self.start = 0  # Points to the oldest item

    def append(self, item):
        item = item.to(self.data.device)
        index = (self.start + self.size) % self.maxlen
        if self.size < self.maxlen:
            self.data[index] = item
            self.size += 1
        else:
            self.data[self.start] = item
            self.start = (self.start + 1) % self.maxlen

    def get(self):
        # Return items in order from oldest to newest
        if self.size < self.maxlen:
            return self.data[:self.size]
        else:
            return torch.cat((self.data[self.start:], self.data[:self.start]), dim=0)

    def __len__(self):
        return self.size
