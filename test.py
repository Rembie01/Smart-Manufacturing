import torch
from nn import Net
from utils import turbine_fft, visualize_freq
from dataset import TurbineDataset
from torch.utils.data import DataLoader

dataset = TurbineDataset(data_dir='Data')
dataloader = DataLoader(dataset, batch_size=1)

time, speed, measurements = next(iter(dataloader))
frequencies, spectrum = turbine_fft(measurements)
spectrum = torch.FloatTensor(spectrum)
spectrum = spectrum.T
spectrum = spectrum[:int(len(spectrum)/2)]
spectrum = spectrum.T

speed = (speed - torch.mean(speed))/torch.std(speed)
measurements = (measurements - torch.mean(measurements))/torch.std(measurements)
spectrum = (spectrum - torch.mean(spectrum))/torch.std(spectrum)
print(spectrum)

print(torch.max(time))
print(speed.shape, torch.min(speed), torch.max(speed), torch.mean(speed))
print(measurements.shape, torch.min(measurements), torch.max(measurements), torch.mean(measurements))
print(spectrum.shape, torch.min(spectrum), torch.max(spectrum), torch.mean(spectrum))

net = Net()
