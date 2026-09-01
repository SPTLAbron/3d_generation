from pathlib import Path
import numpy as np
import trimesh

PROJECT_ROOT=Path(__file__).resolve().parents[2]
MESH_DIR=PROJECT_ROOT/"data"/"meshes"
VOXEL_DIR=PROJECT_ROOT/"data"/"voxels"
RESOLUTION=32

def load_mesh(path):
    mesh=trimesh.load(path,force="mesh")
    if isinstance(mesh,trimesh.Scene): 
        mesh=trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh

def normalize_mesh(mesh):
    mesh=mesh.copy()
    mesh.apply_translation(-mesh.bounding_box.centroid)
    scale=1.8/max(mesh.extents)
    mesh.apply_scale(scale)
    return mesh

def voxelize_mesh(mesh,resolution=RESOLUTION):
    pitch=2.0/resolution
    vox=mesh.voxelized(pitch).fill()
    points=vox.points
    indices=np.floor((points+1.0)/2.0*resolution).astype(int)
    indices=np.clip(indices,0,resolution-1)
    grid=np.zeros((resolution,resolution,resolution),dtype=np.float32)
    grid[indices[:,0],indices[:,1],indices[:,2]]=1.0
    return grid

def voxelize_file(path,resolution=RESOLUTION): 
    return voxelize_mesh(normalize_mesh(load_mesh(path)),resolution)

def main():
    VOXEL_DIR.mkdir(parents=True,exist_ok=True)
    
    for old_voxel in VOXEL_DIR.glob("trophy_*.npy"):
        old_voxel.unlink()
    
    files=sorted(MESH_DIR.glob("*.obj"))
    for i, path in enumerate(files):
        grid=voxelize_file(path)
        np.save(VOXEL_DIR/f"{path.stem}.npy",grid)
        print(f"[{i+1}/{len(files)}] {path.name} -> {grid.shape} occupancy={grid.mean():.4f}")

if __name__=="__main__": main()