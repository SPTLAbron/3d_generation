from pathlib import Path
import sys
import torch
import torch.nn as nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from src.data.dataset import get_loaders
from src.models.autoencoder import Autoencoder3D

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR=ROOT/"outputs"/"checkpoints"

def evaluate(model,loader,loss_fn):
    model.eval(); total=0
    with torch.no_grad():
        for x,_ in loader:
            x=x.to(DEVICE); total+=loss_fn(model(x),x).item()*len(x)
    return total/len(loader.dataset)

def train(epochs=50,batch_size=16,lr=1e-3):
    train_loader,val_loader,_,_,_,_=get_loaders(batch_size)
    model=Autoencoder3D(32).to(DEVICE)
    
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    loss_fn=nn.BCEWithLogitsLoss()
    CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True)
    best_val_loss = float("inf")
    
    for epoch in range(1,epochs+1):
        model.train(); total=0
        for x,_ in train_loader:
            x=x.to(DEVICE)
            optimizer.zero_grad()
            loss=loss_fn(model(x),x)
            loss.backward()
            optimizer.step()
            total+=loss.item()*len(x)
            
        train_loss=total/len(train_loader.dataset)
        val_loss=evaluate(model,val_loader,loss_fn)
        print(f"{epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "ae_best.pt"
            )

            print("  saved new best model")
            
    return model

if __name__=="__main__": train()