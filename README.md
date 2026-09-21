# EgoRecall

EgoRecall is a benchmark for grounding object references in streaming egocentric
observations. This package reads the query tables and annotations, prepares
RGB-D observations from a local ScanNet++ download, and provides query-time
observation access with separate ground-truth supervision.

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

## Read annotations

```python
from pathlib import Path

from egorecall import DatasetPaths
from egorecall.data import EgoRecallAnnotations, decode_program

paths = DatasetPaths.from_toml(Path("configs/paths.toml"))
annotations = EgoRecallAnnotations(paths.dataset_root, split="test", stages=1)

query = next(annotations.iter_queries())
key = (query["scene_id"], query["query_idx"])
assert annotations.get_query(*key) == query

print(query["description"])
print(decode_program(query["program_json"]))
print(annotations.stage_for(*key))
print(annotations.get_frame_name(query["scene_id"], query["frame"]))

# Load full-scene supervision once when inspecting several queries in that scene.
annotation = annotations.get_annotations(query["scene_id"])
target = annotation["objects"][str(query["target_oids"][0])]
print(target["label"], target["visibility_segments"])
```

You can also pass a `Path` directly to `EgoRecallAnnotations`, without a configuration
file. `annotations.query_table` exposes the selected Arrow table for column-oriented
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
  `scene_ids`, `query_keys`, and `len(annotations)` describe this reader's selection.
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

## Prepare ScanNet++ observations

Obtain ScanNet++ under its terms and keep its original directory layout. Each
scene needs these iPhone files for observation preparation:

```text
scannetpp/v2/
  metadata/                       # Optional upstream metadata files
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
The global `metadata/` directory is needed only when requesting a file through
`ScanNetPPScene.metadata_path()`.
Adapted toolkit helpers are documented in
[scannetpp_common/ATTRIBUTION.md](src/scannetpp_common/ATTRIBUTION.md).

Prepare scenes represented in a benchmark selection:

```bash
egorecall-prepare --config configs/paths.toml --split test --stages 1
```

Add `--scenes SCENE_ID` to prepare one scene first. Stages choose which scenes to
prepare; every selected scene retains its complete canonical timeline. Preparation
requires the source pose timeline to match the dataset's frame table exactly.

Preparation reads `manifest.json`, `scenes.json`, and frame rows for the selected
scenes. When `--stages` is supplied, it also reads the scene/split/stage columns
of the stage-assignment table. It does not read query contents or per-scene
visibility annotations. Preparation checks the package format, scene/stage selection,
and selected frame mappings. Use `egorecall-check` for package-wide counts and
query/annotation consistency checks.

Each scene produces `cache_root/<scene_id>.h5`, containing encoded RGB JPEGs,
sensor-depth PNGs, anonymization-mask PNGs, camera matrices, timestamps, and all
source object IDs, labels, oriented boxes, and axis-aligned boxes. It also stores
source-file fingerprints, encoded-frame checksums, and an object-geometry checksum.
RGB is returned as uint8 **RGB**, in native pixel orientation.
Depth is uint16 **millimetres**, with zero representing
invalid depth; depth values are preserved without resizing. Masks retain their
source grayscale values.

The source `aligned_pose` is stored unchanged as a camera-to-world transform in
mesh-aligned coordinates, in metres. Camera axes are x-right, y-down, z-forward;
world Z points up. RGB intrinsics describe the native image grid, and depth
intrinsics scale their first two rows to the 256×192 sensor grid. Use the inverse
of `camera_to_world` when a consumer needs world-to-camera transforms.

Repeated preparation validates the existing cache's scene, timeline, cameras,
object geometry, and source-file hashes before reusing it. Fingerprints include
`segments_anno.json`, so changes to source boxes or labels invalidate reuse.
An incompatible cache raises an error.
New caches are published atomically; interrupted scenes can be prepared again.
FFmpeg's temporary image files use the system temporary directory (configurable
with `TMPDIR`); the temporary H5 is built beside its destination for atomic
publication. Allow temporary space for one scene's selected images.

Scene caches use schema version 2. Earlier caches without object geometry must
be recreated in a new cache directory with the current preparation command.
Normal dataset access uses the EgoRecall annotation package and this prepared
cache; `scannetpp_root` can be omitted after preparation.

To prepare observations without an EgoRecall annotation package, supply the
scene IDs and sampling stride:

```bash
egorecall-prepare --config configs/paths.toml --without-annotations \
  --scenes SCENE_ID --subsample-factor 10
```

This samples sorted pose records every tenth entry, giving a nominal 6 FPS
timeline from the 60 FPS source. With `--without-annotations`, preparation uses
only `scannetpp_root` and `cache_root`; omit `dataset_root` from the configuration.
Both paths use the same observation-preparation process. When an annotation
package is supplied, its frame mapping is also checked against the source
timeline. Sensor timestamps remain available in the cache.

## Read query-time observations

```python
from pathlib import Path

from egorecall import DatasetPaths
from egorecall.data import EgoRecallDataset

dataset = EgoRecallDataset(DatasetPaths.from_toml(Path("configs/paths.toml")), split="test", stages=1)
scene_id, query_idx = dataset.annotations.query_keys[0]

with dataset.open_scene(scene_id) as scene:
    sample = scene.query(query_idx)
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
    answer = scene.answer(query_idx)
    truth = scene.supervision
    print(answer["target_oids"])
    print(len(truth.source_objects), len(truth.filtered_objects))
```

`EgoRecallDataset` is the main entry point. Its `annotations` member is an
`EgoRecallAnnotations` reader, providing query/stage selection and annotation
access. `open_scene()` returns an `EgoRecallScene` context that owns one prepared
scene cache. Geometry is read from that cache, and visibility annotations are
loaded when supervision is requested.

Pass the `QuerySample` to a method. Its query contains only `scene_id`, `query_idx`,
`description`, and `frame`; its observation window includes frame zero through
the query frame. Negative, noninteger, and future-frame indices raise errors.
The window can be iterated, or accessed as encoded images with
`sample.observations.encoded_image(frame_idx, "rgb")`. Keep the scene context open
while using its windows; closing it closes the HDF5 handle.

`sample.observations.camera(frame_idx)` returns a `FrameCamera` containing the
frame index/name, timestamp, pose, and RGB/depth intrinsics. It reads no image
payloads and enforces the same query-time cutoff as `frame()` and `encoded_image()`.
Returned camera arrays are independent copies. Full-timeline tools can use
`SceneH5.camera(frame_idx)` directly; `observation()` includes the same camera
values alongside decoded RGB, depth, and masks.

`scene.supervision` joins full-scene annotations to cached geometry on first
access and retains them for that scene context. `source_objects` contains every ScanNet++
object; `filtered_objects` contains the EgoRecall visibility-filtered population,
joined by `objectId` with matching labels. These are ground truth and include
information unavailable at query time. Observations, answers, and supervision
all work without a configured or accessible raw ScanNet++ directory after preparation.

For direct source access, use `ScanNetPPScene` from `egorecall.data.scannetpp`.
Its `cameras(subsample_factor=10)` returns the full canonical camera sequence,
`objects()` returns all source geometry, and `paths` exposes mesh, segmentation,
annotation, and iPhone filenames. Box axes are stored as rows, and box lengths
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
query/frame/stage joins, and all scene annotations:

```bash
egorecall-check --config configs/paths.toml
```

Also check one scene's raw geometry, source camera timeline, and prepared cache:

```bash
egorecall-check --config configs/paths.toml \
  --source --cache --scenes SCENE_ID --decode-all
```

`--scenes` limits source/cache work; the complete annotation package is always
checked. `--source` validates source camera alignment, object geometry, and
object IDs/labels. Together, `--source --cache` additionally compare source
fingerprints, camera values, and all object geometry against the cache.
These checks use the same source inputs as preparation; they do not require
meshes, segmentation files, or the global metadata directory.
`--cache` verifies cached object structure/checksums and joins object IDs and
labels to the EgoRecall annotations without requiring the raw source. It decodes
the first and last frames by default; `--decode-all` decodes every frame. Each image read checks
its encoded checksum. The JSON report states how many files, scenes, queries,
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
object joins, and observation cutoffs without a dataset download. FFmpeg from the
active environment is required. To also check the reader against a local dataset, run:

```bash
EGORECALL_TEST_DATASET=/path/to/EgoRecall_hf python -m pytest
```

Code uses double quotes, a 120-character line limit, and concrete type annotations.
Docstrings start with a plain descriptive paragraph followed by `Args` and
`Returns` where applicable, without a `Description:` heading. Required data uses
direct access and hard errors; defaults are reserved for genuinely optional
settings and documented behavior.
