![EgoRecall: 3D Visual Grounding from Streaming Egocentric Observations](docs/static/images/social.jpg)

[Hou In Ivan Tam](https://iv-t.github.io/), [Manolis Savva](https://msavva.github.io/) \
Simon Fraser University

[![Project Page](https://img.shields.io/badge/Project%20Page-1d5fb8?style=for-the-badge&logo=data:image%2Fsvg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjggNSA0NiA1NCI%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0iTTkgNmg0M3YxMEgxOXYzMmgzM3YxMEg5eiIvPjxwYXRoIGZpbGw9IiM5Y2MyZWYiIGQ9Ik0xOSAzMlEzNiAxMSA1MyAzMlEzNiA1MyAxOSAzMloiLz48Y2lyY2xlIGN4PSIzNiIgY3k9IjMyIiByPSI1LjUiIGZpbGw9IiMyYjRjN2UiLz48L3N2Zz4=)](https://3dlg-hcvc.github.io/EgoRecall/)
[![Paper](https://img.shields.io/badge/Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)]()
[![Dataset](https://img.shields.io/badge/Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/3dlg-hcvc/EgoRecall)



## Environment Setup
Create and activate the conda environment as follows.
Run the following commands from the root of the repository:
```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate egorecall

# Install the egorecall package
python -m pip install --no-build-isolation -e .
```



## Dataset Setup

### 1. Download the EgoRecall Dataset
First, agree to the terms and request access on [Hugging Face](https://huggingface.co/datasets/3dlg-hcvc/EgoRecall).
Once approved, log in to Hugging Face on your machine ([guide](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)) and download the dataset (about 110 MB):
```bash
# Install the Hugging Face command-line tools
python -m pip install --no-build-isolation -e ".[hub]"

hf auth login
hf download 3dlg-hcvc/EgoRecall --repo-type dataset --local-dir /path/to/EgoRecall_hf
```
The download contains queries, answers, frame mappings, and object visibility annotations for all 183 scenes.


### 2. Download ScanNet++
Our dataset builds on [ScanNet++ v2](https://scannetpp.mlsg.cit.tum.de/scannetpp/), which provides the RGB-D frames, camera poses, and object boxes for each scene.
Request access and download the data following the instructions on the ScanNet++ website.
Keep its original directory layout unchanged.
Each scene needs the following files, which take about 117 GB for the 70 test scenes and 292 GB for all 183 scenes:
```
scannetpp/v2
└── data
    ├── 036bce3393
    │   ├── iphone
    │   │   ├── rgb.mkv
    │   │   ├── rgb_mask.mkv
    │   │   ├── depth.bin
    │   │   ├── pose_intrinsic_imu.json
    │   │   └── exif.json
    │   └── scans
    │       └── segments_anno.json
    ├── ...
```
`configs/benchmark_splits.json` lists the subset of 183 scenes used in EgoRecall and their splits.


### 3. Configure Paths
Copy `configs/paths.example.toml` to `configs/paths.toml` and set the three locations:
```toml
[paths]
dataset_root = "/path/to/EgoRecall_hf"      # the downloaded EgoRecall dataset
scannetpp_root = "/path/to/scannetpp/v2"    # the ScanNet++ folder that contains data/
cache_root = "/path/to/egorecall_cache"     # directory for storing step 4's output
```


### 4. Prepare Observations
Extract relevant data from the ScanNet++ files using our command-line tools.
The prepared files are stored in `cache_root` and are ready for use by our Python API for training and evaluation.

To prepare all splits and check everything afterward, run:
```bash
for s in train val test; do egorecall-prepare --config configs/paths.toml --split $s; done
egorecall-check --config configs/paths.toml --source --cache
```

`egorecall-prepare` reads each scene's ScanNet++ files and packs its sampled frames, cameras, and object boxes into one H5 file in `cache_root`.
```bash
# Prepare all scenes for a split (train, val, or test)
egorecall-prepare --config configs/paths.toml --split test

# Prepare a single scene
egorecall-prepare --config configs/paths.toml --split test --scenes 036bce3393
```

`egorecall-check` checks the validity of the EgoRecall annotations, the ScanNet++ files, and the prepared files.
```bash
# Check the annotations and ScanNet++ files for a split (train, val, or test)
egorecall-check --config configs/paths.toml --source --split test

# Check the prepared files for a split (train, val, or test)
egorecall-check --config configs/paths.toml --cache --split test

# Check the annotations, plus one scene's ScanNet++ files and prepared file
egorecall-check --config configs/paths.toml --source --cache --scenes 036bce3393
```
See `egorecall-prepare --help` and `egorecall-check --help` for more options.

**Storage Requirements**
- Train: 100 scenes, ~31 GB
- Val: 13 scenes, ~4 GB
- Test: 70 scenes, ~23 GB



## Quick Start
Load a query and the frames observed up to its query time:
```python
from pathlib import Path

from egorecall import DatasetPaths
from egorecall.data import EgoRecallDataset

# Select the test queries evaluated in the paper: test stages 1 to 5
dataset = EgoRecallDataset(DatasetPaths.from_toml(Path("configs/paths.toml")), split="test", stages="1:5")
scene_id, query_idx = dataset.annotations.query_keys[0]

with dataset.open_scene(scene_id) as scene_data:
    # Input data: the query text and the frames up to and including the query frame
    sample = scene_data.query(query_idx)
    observation = sample.observations.frame(sample.query.frame)
    print(sample.query.description, len(sample.observations))
    print(observation.rgb.shape, observation.depth.shape, observation.camera_to_world)

    # The ground-truth answer
    answer = scene_data.answer(query_idx)
    print(answer["target_oids"])
```
You can also inspect the query and its answer using our example script:
```bash
python examples/inspect_query.py --config configs/paths.toml --split test --stages 1:5 --scene 036bce3393 --query 15 --supervision
```

**Refer to the [dataset card](https://huggingface.co/datasets/3dlg-hcvc/EgoRecall) for more information on the data format.**


## Evaluation Protocol
Each query is asked at a specific query frame. A method may use the query text and the frames up to and including the query frame; `sample.observations` holds exactly these frames.

The paper reports results on test stages 1–5, which contain 10,000 queries. Validation queries have their own stages, and training queries have none.


## Citation
If you find EgoRecall helpful in your research, please cite our work:
```
@article{tam2026egorecall,
    title = {{EgoRecall}: {3D} Visual Grounding from Streaming Egocentric Observations},
    author = {Tam, Hou In Ivan and Savva, Manolis},
    year = {2026},
    eprint = {XXXX.XXXXX},
    archivePrefix = {arXiv}
}
```
EgoRecall is built on [ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/), so please cite it as well.


## Acknowledgements
This work was funded in part by a Canada Research Chair, NSERC Discovery Grants, and enabled by support from the [Digital Research Alliance of Canada](https://alliancecan.ca/), and an NVIDIA Academic Grant Award.
We thank Austin T. Wang, Denys Iliash, and Weikun Peng for helpful feedback and discussions.
