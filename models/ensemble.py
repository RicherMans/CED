import torch

class EnsembleModel(torch.nn.Module):

    def __init__(self, models) -> None:
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.front_end = self.models[0].front_end

    def forward_spectrogram(self, x):
        ys = []
        for model in self.models:
            y = model.forward_spectrogram(x)
            ys.append(y)
        return torch.stack(ys, dim=-1).mean(-1)

    def forward(self, x):
        ys = []
        for model in self.models:
            y = model(x)
            ys.append(y)
        return torch.stack(ys, dim=-1).mean(-1)
