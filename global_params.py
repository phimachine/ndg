# Set global parameters here in this file
from pathlib import Path

project_root = Path(__file__).parent

# TODO set the data directory and checkpoint directory
# data_source_path = "/home/user/data"
# for your checkpoints and cache
# save_dir = "/home/user/tmp/cache"

data_source_path = "/drive/data"
# for your checkpoints and cache
save_dir = "/drive/Git/cache"

save_dir = Path(save_dir)
data_source_path = Path(data_source_path)
plot_dir = save_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)
