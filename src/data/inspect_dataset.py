import random
from pathlib import Path
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = PROJECT_ROOT / "data" / "meshes"
NUM_TO_INSPECT = 25
SEED = 42

def main():
    obj_files = sorted(MESH_DIR.glob("*.obj"))

    if not obj_files:
        raise RuntimeError(f"No OBJ files found in {MESH_DIR}")

    rng = random.Random(SEED)
    selected = rng.sample(obj_files, min(NUM_TO_INSPECT, len(obj_files)))

    print(f"Found {len(obj_files)} total trophies.")
    print(f"Selected {len(selected)} trophies for inspection.")

    for i, obj_path in enumerate(selected, start=1):
        print(f"[{i}/{len(selected)}] Viewing {obj_path.name}")
        mesh = trimesh.load(obj_path)
        mesh.show()

if __name__ == "__main__":
    main()