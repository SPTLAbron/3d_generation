from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from src.data.dataset import get_loaders
from src.models.autoencoder import Autoencoder3D

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    _, _, _, _, _, test_dataset = get_loaders(batch_size=1)
    model=Autoencoder3D(32).to(DEVICE)
    
    model.load_state_dict(torch.load(ROOT/"outputs"/"checkpoints"/"ae_best.pt",map_location=DEVICE))
    
    model.eval()
    a=test_dataset[0][0].unsqueeze(0).to(DEVICE)
    b=test_dataset[1][0].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        za=model.encode(a); zb=model.encode(b)
        outputs=[torch.sigmoid(model.decode((1-t)*za+t*zb)).cpu().numpy()[0,0] for t in np.linspace(0,1,11)]
    
    out=ROOT/"outputs"/"experiments"/"interpolation"; out.mkdir(parents=True,exist_ok=True)
    
    for i,x in enumerate(outputs): 
        np.save(out/f"{i:02d}.npy",x)

if __name__=="__main__": main()