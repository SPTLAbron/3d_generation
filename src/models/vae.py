import torch
import torch.nn as nn

class VAE3D(nn.Module):
    def __init__(self,latent_dim=32):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv3d(1,32,4,2,1),
            nn.ReLU(),
            nn.Conv3d(32,64,4,2,1),
            nn.ReLU(),
            nn.Conv3d(64,128,4,2,1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.mu=nn.Linear(128*4*4*4,latent_dim); self.logvar=nn.Linear(128*4*4*4,latent_dim)
        self.fc=nn.Linear(latent_dim,128*4*4*4)
        self.decoder=nn.Sequential(
            nn.ConvTranspose3d(128,64,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose3d(64,32,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose3d(32,1,4,2,1),
        )

    def encode(self,x):
        h=self.encoder(x)
        return self.mu(h),self.logvar(h)

    def reparameterize(self,mu,logvar):
        std=torch.exp(.5*logvar)
        return mu+torch.randn_like(std)*std

    def decode(self,z): 
        return self.decoder(self.fc(z).view(-1,128,4,4,4))

    def forward(self,x):
        mu,logvar=self.encode(x); z=self.reparameterize(mu,logvar)
        return self.decode(z),mu,logvar

    def vae_loss(logits,x,mu,logvar,beta=1e-3):
        recon=torch.nn.functional.binary_cross_entropy_with_logits(logits,x,reduction="mean")
        kl=-.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())
        return recon+beta*kl,recon,kl