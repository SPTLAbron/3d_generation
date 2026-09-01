from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

MESH_DIR = DATA_ROOT / "meshes"
VOXEL_DIR = DATA_ROOT / "voxels"
METADATA_DIR = DATA_ROOT / "metadata"

CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
EXPERIMENT_DIR = OUTPUT_ROOT / "experiments"
RENDER_DIR = OUTPUT_ROOT / "renders"

PARAM_COLUMNS = [
    "ball_radius",
    "ball_offset",
    "support_sweep",
    "body_height",
    "body_bottom_radius",
    "body_top_radius",
    "lower_base_radius",
    "lower_base_height",
    "upper_base_radius",
    "upper_base_height",
]

VOXEL_SIZE = 32
LATENT_DIM = 32

TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

RANDOM_SEED = 42

AE_CHECKPOINT = CHECKPOINT_DIR / "ae_best.pt"
VAE_CHECKPOINT = CHECKPOINT_DIR / "vae_best.pt"