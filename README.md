# EgoRecall

EgoRecall is a benchmark for grounding object references in streaming egocentric
observations. This package reads the query tables and annotations, prepares
RGB-D observations from a local ScanNet++ download, and provides query-time
observation access with separate ground-truth supervision.

## Repository layout

| Location | Contents |
|---|---|
| `src/egorecall/data/` | Dataset, annotation, ScanNet++, and H5 readers; table schemas and image codecs |
| `src/egorecall/preparation/` | ScanNet++ to H5 preparation, FFmpeg frame extraction, and depth decoding |
| `src/egorecall/validation/` | Standalone annotation, source, and cache checks |
| `src/egorecall/geometry.py` | Shared camera and box records and intrinsic scaling |
| `src/egorecall/arguments.py`, `src/egorecall/integrity.py` | Shared value checks, file/path helpers, and fingerprints |
| `src/egorecall/cli/` | Command-line entry points |
| `ATTRIBUTION.md` | Code adapted from the ScanNet++ toolkit |
| `tests/` | Data, preparation, validation, and integration tests with shared fixtures |
| `examples/` | Small examples using the public API |
| `docs/` | GitHub Pages project website |

## Install

From the repository root, create a Python 3.12 environment and install the package:

```bash
mamba env create --file environment.yml
mamba activate egorecall
python -m pip install --no-build-isolation -e .
```

The named environment is installed in mamba's default environment directory.
`conda env create` can be used in place of `mamba env create`. The environment
includes PyArrow, NumPy, HDF5/image/depth libraries, FFmpeg, and development tools.
These workflows run on the CPU and do not require rendering or perception models.
For an existing environment, run `mamba env update --file environment.yml`,
activate it, and repeat the editable installation above.

When installing through pip in another Python 3.12 environment, install FFmpeg
6 or newer separately and ensure `ffmpeg` is on `PATH`. `egorecall-prepare`
also accepts `--ffmpeg /path/to/ffmpeg`.

## Download annotations

Request access at [3dlg-hcvc/EgoRecall](https://huggingface.co/datasets/3dlg-hcvc/EgoRecall).
After approval, install the optional Hub tools and authenticate:

```bash
python -m pip install --no-build-isolation -e ".[hub]"
hf auth login
hf download 3dlg-hcvc/EgoRecall --repo-type dataset --local-dir /path/to/EgoRecall_hf
```

This downloads queries, answers, frame mappings, and object visibility annotations.
Obtain ScanNet++ v2 separately for the observations and geometry used below.
The data-access package does not include baseline runners or the official scorer.

## Configure paths

Copy `configs/paths.example.toml` to `configs/paths.toml` and edit the locations:

```toml
[paths]
dataset_root = "/path/to/EgoRecall_hf"
scannetpp_root = "/path/to/scannetpp/v2"
cache_root = "/path/to/egorecall_cache"
```

`dataset_root` identifies the EgoRecall data directory. `scannetpp_root` identifies
the ScanNet++ root containing `data/`, and `cache_root` identifies
a separate directory for prepared assets. Do not point `scannetpp_root` directly
at its `data/` subtree.

Configure only the roots used by the operation. Annotation loading/checking
requires `dataset_root`; scene-cache access additionally requires `cache_root`.
Preparation requires `scannetpp_root` and `cache_root`, plus `dataset_root` when
using annotations. `--source` checking requires `scannetpp_root`.
`DatasetPaths` converts the configured roots to absolute paths; those
directories need not exist, and their access permissions are not validated.
Relative paths resolve from the configuration file's directory, independent of
the calling directory.

## Check before use

Run the standalone checker after downloading the annotations:

```bash
egorecall-check --config configs/paths.toml
```

Before preparation, add `--source --scenes SCENE_ID` to check a scene's source
files, cameras, boxes, and agreement with the annotations. After preparation,
run with `--cache --scenes SCENE_ID` to check the H5 file. Use `--source --cache`
together to also compare the cache with the current source files.

Readers assume the inputs have been checked. They do not repeat schema, count,
geometry, or checksum audits while loading queries and frames. Missing files,
keys, and out-of-range accesses raise at the operation that uses them. Run the
checker again after replacing or modifying annotation, source, or cache files.

## Read annotations

```python
from pathlib import Path

from egorecall import DatasetPaths
from egorecall.data import EgoRecallAnnotations, decode_program

paths = DatasetPaths.from_toml(Path("configs/paths.toml"))
annotation_reader = EgoRecallAnnotations(paths.dataset_root, split="test", stages=1)

query = next(annotation_reader.iter_queries())
key = (query["scene_id"], query["query_idx"])
assert annotation_reader.get_query(*key) == query

print(query["description"])
print(decode_program(query["program_json"]))
print(annotation_reader.stage_for(*key))
print(annotation_reader.get_frame_name(query["scene_id"], query["frame"]))

# Load full-scene supervision once when inspecting several queries in that scene.
scene_annotations = annotation_reader.get_annotations(query["scene_id"])
target = scene_annotations["objects"][str(query["target_oids"][0])]
print(target["label"], target["visibility_segments"])
```

You can also pass a `Path` directly to `EgoRecallAnnotations`, without a configuration
file. `annotation_reader.query_table` exposes the selected Arrow table for column-oriented
processing; dictionaries are produced in bounded batches by `iter_queries()`.

### Query identity and selection

- `(scene_id, query_idx)` identifies a query in both the query and stage tables. IDs can have gaps and stay
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
  `scene_ids`, `query_keys`, and `len(annotation_reader)` describe this reader's selection.
- Scene metadata counts from `get_scene()` cover all stored rows for the scene,
  before stage filtering. `get_query()` and `stage_for()` only accept keys
  in the reader's selection.

Stage selection filters the query Parquet read by scene/query IDs, so a small
selection does not first create Python records for the full split.
The reader preserves query-table order. Frame mappings are ordered by their
explicit `frame_idx`, even if the frame table's storage order changes.

`configs/benchmark_splits.json` records the benchmark's fixed assignment of all
183 scenes: 100 train, 13 validation, and 70 test. It includes the assignment seed
and source checksum. The reader determines availability from the data directory;
the assignment file documents membership across the full benchmark.

### Annotations and observation boundaries

The query `frame` and frame-table `frame_idx` count sampled images from zero.
For example, with a stride of 10, frame 1 refers to source image `frame_000010`. Object IDs are scoped to their scene. Visibility segment endpoints are
inclusive, and `per_frame` keys are frame indices encoded as JSON strings.

Answers, DSL programs, and visibility annotations are supervision. Scene
annotations include every filtered object and the entire scene history, including
frames after individual queries. They must not be supplied as model observations
at query time. The nominal timeline is 6 FPS; frame names come from the frame
table and exact timestamps come from ScanNet++ camera metadata.

### Validation and supported layout

`egorecall-check` checks table schemas, duplicate IDs, query/stage assignments,
scene counts, frame numbers, query programs and answers, and every scene's object
visibility file. It also checks the file checksums listed in the manifest.
The readers load these same files without repeating the checks.

The dataset directory contains train, val, and test splits. Its schema-2
manifest declares per-split counts and stage ranges, plus aggregate counts.

`counts.stage_assignments` counts rows in the stage-assignment table, with one
assignment per validation/test query and zero for unstaged training. For example,
a dataset containing the 2,000 queries in stage 1 has 2,000 stage assignments.

The directory layout is `queries/<split>.parquet`,
`stages/<val-or-test>.parquet`, `frames/<split>.parquet`, `scenes.json`,
`annotations/<scene_id>.json.gz`, and `manifest.json`, with one Parquet file per
table. The checker validates the stored visibility statistics and their frame
references; it does not rerun visibility rendering from source geometry.

## Prepare ScanNet++ observations

Obtain ScanNet++ under its terms and keep its original directory layout. Each
scene needs these iPhone files for observation preparation:

```text
scannetpp/v2/
  data/<scene_id>/
    iphone/
      rgb.mkv
      rgb_mask.mkv
      depth.bin
      pose_intrinsic_imu.json
      exif.json
    scans/
      mesh_aligned_0.05.ply
      segments.json
      segments_anno.json
```

Preparation reads the iPhone files and `scans/segments_anno.json` to cache
observations and all source object IDs, labels, and boxes. Mesh and segmentation
files support operations such as visibility rendering; they are not needed to
read a prepared scene's observations or supervision. DSLR assets and COLMAP
reconstruction files are not required for preparation.
Code adapted from the ScanNet++ toolkit is listed in [ATTRIBUTION.md](ATTRIBUTION.md).

Check the raw inputs, then prepare scenes represented in a benchmark selection:

```bash
egorecall-check --config configs/paths.toml --source --scenes SCENE_ID
egorecall-prepare --config configs/paths.toml --split test --stages 1 --scenes SCENE_ID
egorecall-check --config configs/paths.toml --cache --scenes SCENE_ID
```

Omit `--scenes` to prepare every scene represented in the requested stages.
Stages select scenes; every selected scene retains its complete sampled frame
sequence. The source check verifies that these frame names agree with the
annotation frame table. `egorecall-check` accepts the same `--split` and
`--stages` options to check every scene in a selection:

```bash
egorecall-check --config configs/paths.toml --cache --split test --stages 1
```

Preparation selects scenes with `EgoRecallAnnotations`, the same reader as the
Python API, and takes each scene's sampling settings and frame names from it.
Object visibility files are not read during preparation.

Each scene produces `cache_root/<scene_id>.h5`, containing encoded RGB JPEGs,
sensor-depth PNGs, anonymization-mask PNGs, camera matrices, timestamps, and all
source object IDs, labels, oriented boxes, and axis-aligned boxes. It also stores
source-file fingerprints, encoded-frame checksums, and camera and object-geometry checksums.
RGB is returned as uint8 **RGB**, in native pixel orientation.
Depth is uint16 **millimetres**, with zero representing
invalid depth; depth values are preserved without resizing. Masks retain their
source grayscale values.

The source `aligned_pose` is stored unchanged as a camera-to-world transform in
mesh-aligned coordinates, in metres. Camera axes are x-right, y-down, z-forward;
world Z points up. RGB intrinsics describe the native image grid, and depth
intrinsics scale their first two rows to the 256×192 sensor grid. Use the inverse
of `camera_to_world` when a consumer needs world-to-camera transforms.

Repeated preparation leaves an existing scene cache in place. To detect stale
caches, run `egorecall-check --source --cache`: it compares source-file hashes,
cameras, and object boxes with the cache. Use a new cache directory when preparing
changed inputs or sampling settings.
New caches are published atomically; interrupted scenes can be prepared again.
FFmpeg's temporary image files use the system temporary directory (configurable
with `TMPDIR`); the temporary H5 is built beside its destination for atomic
publication. Allow temporary space for one scene's selected images.

Scene caches use schema version 3 and include object geometry. Normal dataset
access uses the EgoRecall annotation package and this prepared cache;
`scannetpp_root` can be omitted after preparation.

To prepare observations without an EgoRecall annotation package, supply the
scene IDs and sampling stride:

```bash
egorecall-check --config configs/paths.toml --without-annotations \
  --scenes SCENE_ID --subsample-factor 10
egorecall-prepare --config configs/paths.toml --without-annotations \
  --scenes SCENE_ID --subsample-factor 10
egorecall-check --config configs/paths.toml --without-annotations \
  --scenes SCENE_ID --subsample-factor 10 --cache
```

This samples sorted pose records every tenth entry, giving a nominal 6 FPS
timeline from the 60 FPS source. With `--without-annotations`, preparation uses
only `scannetpp_root` and `cache_root`; omit `dataset_root` from the configuration.
Both paths use the same observation-preparation process. Source checks compare
frame names with the annotation package when it is supplied. Sensor timestamps
remain available in the cache.

## Read query-time observations

```python
from pathlib import Path

from egorecall import DatasetPaths
from egorecall.data import EgoRecallDataset

dataset = EgoRecallDataset(DatasetPaths.from_toml(Path("configs/paths.toml")), split="test", stages=1)
scene_id, query_idx = dataset.annotations.query_keys[0]

with dataset.open_scene(scene_id) as scene_data:
    sample = scene_data.query(query_idx)
    print(sample.query.description)
    print(len(sample.observations))

    # Only frames through the query time are accessible from this sample.
    observation = sample.observations.frame(sample.query.frame)
    print(observation.rgb.shape, observation.depth.dtype)
    print(observation.camera_to_world, observation.depth_intrinsics)

    # Read just camera metadata when images are not needed.
    camera = sample.observations.camera(sample.query.frame)
    print(camera.timestamp, camera.camera_to_world)

    # Request ground truth separately for inspection or evaluation.
    answer = scene_data.answer(query_idx)
    scene_supervision = scene_data.supervision
    print(answer["target_oids"])
    print(len(scene_supervision.source_objects), len(scene_supervision.filtered_objects))
```

`EgoRecallDataset` is the main entry point. Its `annotations` member is an
`EgoRecallAnnotations` reader, providing query/stage selection and annotation
access. `open_scene()` returns an `EgoRecallScene` context that owns one prepared
scene cache; the scene must have at least one query in the selection. Geometry is
read from that cache, and visibility annotations are loaded when supervision is
requested.

Pass the `QuerySample` to a method. Its query contains only `scene_id`, `query_idx`,
`description`, and `frame`; its observation window includes frame zero through
the query frame. Noninteger indices and indices outside that window raise errors.
Negative indices count backward from the query frame: `frame(-1)` is the query
frame. The window can be iterated, or accessed as encoded images with
`sample.observations.encoded_image(frame_idx, "rgb")`. Keep the scene context open
while using its windows; closing it closes the HDF5 handle.

`sample.observations.camera(frame_idx)` returns a `FrameCamera` containing the
frame index/name, timestamp, pose, and RGB/depth intrinsics. It reads no image
payloads and enforces the same query-time cutoff as `frame()` and `encoded_image()`.
Returned camera arrays are independent copies. Full-timeline tools can use
`SceneH5.camera(frame_idx)` directly; `observation()` includes the same camera
values alongside decoded RGB, depth, and masks.

`scene_data.supervision` reads visibility histories and cached object boxes on first
access and retains them for that scene context. `source_objects` contains every ScanNet++
object in the scene; `filtered_objects` contains the subset whose IDs appear
in the visibility annotations. These are ground truth and include
information unavailable at query time. Observations, answers, and supervision
all work without a configured or accessible raw ScanNet++ directory after preparation.

For direct source access, use `ScanNetPPScene` from `egorecall.data.scannetpp`.
Its `cameras(subsample_factor=10)` returns the camera records for every sampled frame,
`objects()` returns all source geometry, and attributes such as `iphone_video_path`
and `scan_anno_json_path` locate the source files that preparation reads. Box axes are stored as rows, and box lengths
are full side lengths in metres. `SceneH5` from `egorecall.data.scene_h5` provides
full-timeline cache access and an `objects()` method for cached IDs, labels, and
boxes. Object records returned by `objects()` own their geometry arrays, so
editing them does not modify later lookups.

The inspection example combines these operations for a chosen stable query key:

```bash
python examples/inspect_query.py --config configs/paths.toml \
  --split test --stages 1 --scene SCENE_ID --query QUERY_IDX --supervision
```

## Check data and caches

Verify every package file listed in the integrity manifest, required-file coverage,
query and stage IDs, frame mappings, and all scene annotations:

```bash
egorecall-check --config configs/paths.toml
```

Also check one scene's raw geometry, source camera timeline, and prepared cache:

```bash
egorecall-check --config configs/paths.toml \
  --source --cache --scenes SCENE_ID --decode-all
```

`--split`, `--stages`, and `--scenes` limit source/cache work. With `--split`, scenes are
selected as `egorecall-prepare` selects them; `--scenes` alone may name scenes from any
split. The complete annotation package is always checked.
`--source` validates source camera alignment, object geometry, and
object IDs/labels. Together, `--source --cache` additionally compare source
fingerprints, camera values, and all object geometry against the cache.
These checks use the same source inputs as preparation; they do not require
meshes or segmentation files.
`--cache` verifies the cached camera and object checksums and structure, and compares
object IDs and labels with the EgoRecall annotations without requiring the raw source. It decodes
the first and last frames by default; `--decode-all` decodes every frame.
Every cache check verifies all encoded image checksums and headers, including
frames that are not fully decoded. Normal image reads do not repeat these checks. The JSON report states how many files, scenes, queries,
and frames were checked. Missing files or mismatches produce errors.

The commands are also available as `python -m egorecall.cli.prepare_scannetpp`
and `python -m egorecall.cli.check_dataset`.

## Development checks

```bash
python -m pytest
ruff check .
ruff format --check .
```

Synthetic fixtures exercise selection, source decoding, cache reuse/integrity,
matching object IDs/labels, and observation cutoffs without a dataset download. FFmpeg from the
active environment is required. To also check the reader against a local dataset, run:

```bash
EGORECALL_TEST_DATASET=/path/to/EgoRecall_hf python -m pytest
```

Code uses double quotes, a 120-character line limit, and concrete type annotations.
Docstrings start with a plain descriptive paragraph followed by `Args` and
`Returns` where applicable, without a `Description:` heading. Required data uses
direct access and hard errors; defaults are reserved for genuinely optional
settings and documented behavior.

## Licenses

Original EgoRecall code is distributed under the [MIT license](LICENSE),
Copyright (c) 2026 3dlg-hcvc. ScanNet++ toolkit adaptations are identified in
[ATTRIBUTION.md](ATTRIBUTION.md). EgoRecall dataset
annotations use CC BY-NC 4.0 as stated in the dataset's own license and card.
For code, dataset, or access questions, use
[GitHub Issues](https://github.com/3dlg-hcvc/EgoRecall/issues).
