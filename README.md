# EgoRecall

EgoRecall is a benchmark for grounding object references in streaming egocentric
observations. This package reads the query tables, stage assignments, frame
mappings, and object visibility annotations that define the dataset.

## Install

From the repository root, create a Python 3.12 environment and install the package:

```bash
mamba env create --file environment.yml
mamba activate egorecall
python -m pip install --no-build-isolation -e .
```

The named environment is installed in mamba's default environment directory.
`conda env create` can be used in place of `mamba env create`. The environment
includes PyArrow and development tools for annotation loading and validation.
No model or GPU dependencies are required for these operations.

## Configure paths

Copy `configs/paths.example.toml` to `configs/paths.toml` and edit the locations:

```toml
[paths]
dataset_root = "/path/to/EgoRecall_hf"
scannetpp_root = "/path/to/scannetpp/v2"
cache_root = "/path/to/egorecall_cache"
```

`dataset_root` identifies the EgoRecall data directory. `scannetpp_root` identifies
the ScanNet++ download, and `cache_root` identifies a directory for prepared assets.
The annotation reader uses only `dataset_root`, so the other two settings can be
omitted. `DatasetPaths` converts the configured roots to absolute paths; those
directories need not exist, and their access permissions are not validated.
Relative paths resolve from the configuration file's directory, independent of
the calling directory.

## Read the local dataset

```python
from pathlib import Path

from egorecall import DatasetPaths
from egorecall.data import EgoRecallDataset, decode_program

paths = DatasetPaths.from_toml(Path("configs/paths.toml"))
dataset = EgoRecallDataset(paths.dataset_root, split="test", stages=1)

query = next(dataset.iter_queries())
key = (query["scene_id"], query["query_idx"])
assert dataset.get_query(*key) == query

print(query["description"])
print(decode_program(query["program_json"]))
print(dataset.stage_for(*key))
print(dataset.get_frame_name(query["scene_id"], query["frame"]))

# Load full-scene supervision once when inspecting several queries in that scene.
annotation = dataset.get_annotations(query["scene_id"])
target = annotation["objects"][str(query["target_oids"][0])]
print(target["label"], target["visibility_segments"])
```

You can also pass a `Path` directly to `EgoRecallDataset`, without a configuration
file. `dataset.query_table` exposes the selected Arrow table for column-oriented
processing; dictionaries are produced in bounded batches by `iter_queries()`.

### Query identity and selection

- `(scene_id, query_idx)` is the stable join key. IDs can have gaps and stay
  unchanged across stage selections.
- `source_query_id` is the identifier assigned when a query record is generated,
  retained through balancing for provenance. `program_depth` is DSL nesting depth.
- A stage is a predefined group of evaluation queries. `stages=3` selects
  **stage 3 only**, and `stages="1:5"` selects stages 1 through 5.
  `stages=None` selects all rows in the split's query table.
- The benchmark's 10,000-query evaluation subset is test stages 1–5. The reader
  raises an error if any requested stage is absent from the data directory.
- Training is unstaged. Omit `stages` when reading a training dataset.
- `available_stages` and `available_splits` describe the data in the dataset directory;
  `scene_ids`, `query_keys`, and `len(dataset)` describe this reader's selection.
- Scene metadata counts from `get_scene()` cover all stored rows for the scene,
  before stage filtering. `get_query()` and `stage_for()` only accept keys
  in the reader's selection.

The reader preserves query-table order. Frame mappings are ordered by their
explicit `frame_idx`, even if the frame table's storage order changes.

`configs/benchmark_splits.json` records the benchmark's fixed assignment of all
183 scenes: 100 train, 13 validation, and 70 test. It includes the assignment seed
and source checksum. The reader determines availability from the data directory;
the assignment file documents membership across the full benchmark.

### Annotations and observation boundaries

The query `frame` and frame-table `frame_idx` are zero-based canonical timeline
indices. Object IDs are scoped to their scene. Visibility segment endpoints are
inclusive, and `per_frame` keys are frame indices encoded as JSON strings.

Answers, DSL programs, and visibility annotations are supervision. Scene
annotations include every filtered object and the entire scene history, including
frames after individual queries. They must not be supplied as model observations
at query time. The nominal timeline is 6 FPS; frame names come from the frame
table and exact timestamps come from ScanNet++ camera metadata.

### Validation and supported layout

The reader checks table schemas, duplicate keys, split and stage membership,
scene counts, frame alignment/bounds, and selected query structure/target
partitions. `get_annotations()` additionally checks annotation field types,
scene/frame identity, object counts, and target-object joins. Annotation files
are loaded when `get_annotations()` is called.

The reader supports a data directory for one split, whose manifest explicitly
declares its split, counts, and (for validation/test) stage range. Missing required
fields and mismatched declarations raise errors.

`counts.stage_assignments` counts rows in the stage-assignment table, with one
assignment per validation/test query and zero for unstaged training. For example,
a dataset containing the 2,000 queries in stage 1 has 2,000 stage assignments.

The directory layout is `queries/<split>.parquet`,
`stages/<val-or-test>.parquet`, `frames/<split>.parquet`, `scenes.json`,
`annotations/<scene_id>.json.gz`, and `manifest.json`, with one Parquet file per
table. Loading validates data structure and joins; it does not verify all file
checksums or recompute visibility statistics.

## Development checks

```bash
python -m pytest
ruff check .
ruff format --check .
```

Synthetic fixtures exercise selection, joins, and invalid inputs without a dataset
download. To also check the reader against a local dataset, run:

```bash
EGORECALL_TEST_DATASET=/path/to/EgoRecall_hf python -m pytest
```

Code uses double quotes, a 120-character line limit, and concrete type annotations.
Docstrings start with a plain descriptive paragraph followed by `Args` and
`Returns` where applicable, without a `Description:` heading. Required data uses
direct access and hard errors; defaults are reserved for genuinely optional
settings and documented behavior.
