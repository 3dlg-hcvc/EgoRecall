# ScanNet++ toolkit attribution

Parts of this package are adapted from the official
[ScanNet++ toolkit](https://github.com/scannetpp/scannetpp):

| File | Upstream reference | Adaptation |
|---|---|---|
| `src/egorecall/preparation/depth.py` | Depth extraction in `iphone/prepare_iphone_data.py` | Supports the same whole-stream DEFLATE and per-frame LZ4/DEFLATE formats, with bounded decoding, explicit errors, and uint16 millimetre output. |
| `src/egorecall/data/scannetpp.py` | `common/scene_release.py`; `load_annotation()` in `common/utils/anno.py` | `ScanNetPPScene` retains the scan/iPhone path conventions for the files used to prepare observations, and indexes source objects by `objectId`. Other accessors, rendering, GPU imports, and vertex assignment are omitted. |

Reference toolkit commit: `d5e644913a4721a3a02f07cd909e5d62d4c2432f`.
