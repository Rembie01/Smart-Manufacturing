import torch as T


class AutoEncoder(T.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = T.nn.Sequential(
            T.nn.Linear(8192, 4096),
            T.nn.Tanh(),
            T.nn.Linear(4096, 2048),
            T.nn.Tanh(),
            T.nn.Linear(2048, 1024),
            T.nn.Tanh(),
            T.nn.Linear(1024, 512),
            T.nn.Tanh(),
            T.nn.Linear(512, 256),
            T.nn.Tanh(),
            T.nn.Linear(256, 128),
            T.nn.Tanh(),
            T.nn.Linear(128, 64),
            T.nn.Tanh(),
            T.nn.Linear(64, 32),
            T.nn.Tanh(),
            T.nn.Linear(32, 16),
        )
        self.dec = T.nn.Sequential(
            T.nn.Linear(16, 32),
            T.nn.Tanh(),
            T.nn.Linear(32, 64),
            T.nn.Tanh(),
            T.nn.Linear(64, 128),
            T.nn.Tanh(),
            T.nn.Linear(128, 256),
            T.nn.Tanh(),
            T.nn.Linear(256, 512),
            T.nn.Tanh(),
            T.nn.Linear(512, 1024),
            T.nn.Tanh(),
            T.nn.Linear(1024, 2048),
            T.nn.Tanh(),
            T.nn.Linear(2048, 4096),
            T.nn.Tanh(),
            T.nn.Linear(4096, 8192),
        )

    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode
