from utils import turbine_fft, visualize_freq
from dataset import TurbineDataset
from torch.utils.data import DataLoader

dataset = TurbineDataset(data_dir='Data')
dataloader = DataLoader(dataset)

_, _, measurements = next(iter(dataloader))

for row in measurements:
    freq, fft = turbine_fft(row)
    visualize_freq(freq, fft)
