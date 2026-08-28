import torch
import torch.nn as nn

class Autoencoder3D(nn.Module):
    def __init__(self,latent_dim=32):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv3d(1,32,4,2,1),nn.ReLU(),nn.Conv3d(32,64,4,2,1),nn.ReLU(),nn.Conv3d(64,128,4,2,1),nn.ReLU(),nn.Flatten())
        self.to_latent=nn.Linear(128*4*4*4,latent_dim)
        self.from_latent=nn.Linear(latent_dim,128*4*4*4)
        self.decoder=nn.Sequential(nn.ConvTranspose3d(128,64,4,2,1),nn.ReLU(),nn.ConvTranspose3d(64,32,4,2,1),nn.ReLU(),nn.ConvTranspose3d(32,1,4,2,1))

    def encode(self,x): 
        return self.to_latent(self.encoder(x))

    def decode(self,z):
        x=self.from_latent(z).view(-1,128,4,4,4)
        return self.decoder(x)

    def forward(self,x): 
        return self.decode(self.encode(x))

if __name__=="__main__":
    model=Autoencoder3D()
    x=torch.randn(2,1,32,32,32)
    z=model.encode(x)
    y=model(x)
    print(x.shape,z.shape,y.shape)