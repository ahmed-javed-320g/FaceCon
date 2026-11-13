# Face model attribution

The realistic face geometry used by this prototype is derived from the
**ICT-FaceKit** face model.

- Source: https://github.com/ICT-VGL/ICT-FaceKit
- Licence: MIT License (permissive; compatible with this prototype)
- What we use: the generic neutral mesh and the leading identity morph targets
  (`identity000.obj` … `identityNNN.obj`). The identity basis is formed as
  `identity_k - neutral`, packaged by `scripts/build_face_model.py`.

This serves as a licence-clean, realistic stand-in for FLAME while controlled
3DFN (dbGaP + FaceBase) access is pending. The pipeline consumes it only through
the shape vector `beta` and `FaceModel.decode`, so swapping in real FLAME later
is a single-module change.

To (re)build the model from scratch:

    python scripts/build_face_model.py 50
