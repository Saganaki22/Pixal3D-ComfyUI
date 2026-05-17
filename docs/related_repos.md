# Related Repo Notes

This is a short map of useful nearby work. These repos are references for wheels and export behavior, not required runtime dependencies.

## visualbruno/ComfyUI-Trellis2

Repo: https://github.com/visualbruno/ComfyUI-Trellis2

Useful part: the Windows wheel folder includes builds for `cumesh`, `flex_gemm`, `o_voxel`, `nvdiffrast`, `nvdiffrec_render`, and some NATTEN wheels.

Wheel folder:

```text
https://github.com/visualbruno/ComfyUI-Trellis2/tree/main/wheels
```

Important limitation: do not install a wheel just because the Python tag matches. It also has to match the Torch and CUDA ABI.

## ThatButters/trellis2-blackwell-fix

Repo: https://github.com/ThatButters/trellis2-blackwell-fix

Useful part: documents a CuMesh remesh failure mode where export can create fragmented meshes instead of crashing.

Pixal3D-ComfyUI exposes `remesh` on **Pixal3D Export GLB** so users can turn that o_voxel remesh path on or off directly. If exported GLBs look shredded, keep `decimation_target=1000000` or higher and try `remesh=true`.
