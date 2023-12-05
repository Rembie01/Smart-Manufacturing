import datetime

import torch
from nn import AutoEncoder
from utils import turbine_fft, visualize_freq
from dataset import TurbineDataset
from torch.utils.data import DataLoader

import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import timedelta

model = r'D:\PyCharm\Smart-Manufacturing\Autoencoder_05-12-2023_15-19-19.pth'

lr = 1e-2         # learning rate
w_d = 1e-5        # weight decay
epochs = 30

dataset = TurbineDataset(data_dir='Data', type='train')
dataloader_0 = DataLoader(dataset, batch_size=1000)

for i, (_, _, data) in enumerate(dataloader_0):
    frequencies, spectrum = turbine_fft(data)
    spectrum = torch.FloatTensor(spectrum)
    spectrum = spectrum.T
    spectrum = spectrum[:int(len(spectrum) / 2)]  # delete upper half of spectrum
    spectrum = spectrum.T
    train_mean = torch.mean(spectrum)
    train_std = torch.std(spectrum)
    print(train_mean, "+-", train_std)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
net = AutoEncoder()

metrics = defaultdict(list)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
net.to(device)

if model is not None:
    net.load_state_dict(torch.load(model))

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=w_d)

net.train()
start = time.time()
for epoch in range(epochs):
    ep_start = time.time()
    running_loss = 0.0
    for i, (_, speed, data) in enumerate(dataloader):
        frequencies, spectrum = turbine_fft(data)

        spectrum = torch.FloatTensor(spectrum)
        spectrum = spectrum.T
        spectrum = spectrum[:int(len(spectrum) / 2)]  # delete upper half of spectrum
        spectrum = spectrum.T
        spectrum_values = (spectrum - train_mean) / train_std

        sample = net(spectrum_values.to(device))
        loss = criterion(spectrum_values.to(device), sample)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss/len(dataloader)
    metrics['train_loss'].append(epoch_loss)
    ep_end = time.time()
    print('-----------------------------------------------')
    print('[EPOCH] {}/{}\n[LOSS] {}'.format(epoch+1, epochs, epoch_loss))
    print('Epoch Complete in {}'.format(timedelta(seconds=ep_end-ep_start)))
end = time.time()
print('-----------------------------------------------')
print('[System Complete: {}]'.format(timedelta(seconds=end-start)))

_, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.set_title('Loss')
ax.plot(metrics['train_loss'])
plt.show()

net.eval()
loss_dist_train = []
loss_dist_test = []
times = []

for i, (time, speed, data) in enumerate(dataloader):
    frequencies, spectrum = turbine_fft(data)
    spectrum = torch.FloatTensor(spectrum)
    spectrum = spectrum.T
    spectrum = spectrum[:int(len(spectrum) / 2)]  # delete upper half of spectrum
    spectrum = spectrum.T
    spectrum_values = (spectrum - train_mean) / train_std

    sample = net(spectrum_values.to(device))
    loss = criterion(spectrum_values.to(device), sample)
    loss_dist_train.append(loss.item())

dataset2 = TurbineDataset(data_dir='Data', type='test')
dataloader2 = DataLoader(dataset2, batch_size=1)

for i, (time, speed, data) in enumerate(dataloader2):
    frequencies2, spectrum2 = turbine_fft(data)
    spectrum2 = torch.FloatTensor(spectrum2)
    spectrum2 = spectrum2.T
    spectrum2 = spectrum2[:int(len(spectrum2) / 2)]  # delete upper half of spectrum
    spectrum2 = spectrum2.T
    spectrum2 = (spectrum2 - train_mean) / train_std

    sample = net(spectrum2.to(device))
    loss = criterion(spectrum2.to(device), sample)
    loss_dist_test.append(loss.item())
    times.append(time)

lower_threshold = 0.0
upper_threshold = max(loss_dist_train)
plt.figure(figsize=(12, 6))
plt.title('Loss Distribution')
sns.displot(loss_dist_train, bins=100, kde=True, color='green')
sns.displot(loss_dist_test, bins=100, kde=True, color='blue')
plt.axvline(upper_threshold, 0.0, 10, color='r')
plt.show()

torch.save(net.state_dict(), "D:/PyCharm/Smart-Manufacturing/Autoencoder_" + datetime.datetime.utcnow().strftime("%d-%m-%Y_%H-%M-%S") + ".pth")

for i, value in enumerate(loss_dist_test):
    if value > upper_threshold:
        print("Faulty:", times[i].item())
