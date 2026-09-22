# ScanNet++ toolkit attribution

These helpers are adapted from the official
[ScanNet++ toolkit](https://github.com/scannetpp/scannetpp):

| File | Upstream reference | Adaptation |
|---|---|---|
| `scene_release.py` | `common/scene_release.py` | Retains the scan/iPhone path conventions with typed paths relative to the data subtree. Unused DSLR and panorama accessors are omitted. |
| `annotations.py` | `load_annotation()` in `common/utils/anno.py` | Retains objectId-based indexing with typed records. Rendering, GPU imports, and vertex assignment are omitted. |
| `iphone.py` | Depth extraction in `iphone/prepare_iphone_data.py` | Supports the same whole-stream DEFLATE and per-frame LZ4/DEFLATE formats, with bounded decoding, explicit errors, and uint16 millimetre output. |

Reference toolkit commit: `d5e644913a4721a3a02f07cd909e5d62d4c2432f`.
