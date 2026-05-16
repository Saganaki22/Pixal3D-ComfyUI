# Building NATTEN on Windows

Guide for building NATTEN from source on Windows with CUDA.

## Prerequisites

- **Python 3.11–3.13** (recommended: [python.org](https://python.org) installer — Microsoft Store Python is missing `python3X.lib`)
- **CUDA Toolkit 12.8–13.0** — [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- **Visual Studio 2022 Build Tools** with "Desktop development with C++" workload — [visualstudio.microsoft.com](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- **Git**

## Compute Capability Reference

| GPU Series | Architecture | CUDA_ARCH |
|---|---|---|
| GTX 1650–1660 Ti | Turing | 7.5 |
| RTX 2060–2080 Ti | Turing | 7.5 |
| Titan RTX | Turing | 7.5 |
| RTX 3050–3090 Ti | Ampere | 8.6 |
| RTX 4050–4090 | Ada Lovelace | 8.9 |
| RTX 5050–5090 | Blackwell | 12.0 |

Not sure? Run: `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`

> **If your GPU is not listed, it is not supported.** NATTEN requires compute capability 7.5+.

## Build Steps

### 1. Create a virtual environment

```bat
python -m venv C:\natten-env
C:\natten-env\Scripts\activate
```

### 2. Install dependencies

```bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

> Replace `cu130` with your CUDA version: `cu128`, `cu130`, etc.

```bat
pip install ninja cmake wheel "setuptools>=64,<69"
```

> `setuptools>=69` may reject `where = ["src"]` in `pyproject.toml`. If you get an error about `src/`, either pin setuptools < 69 or ensure `pyproject.toml` has `where = ["src"]` (no trailing slash).

### 3. Clone NATTEN

```bat
git clone --recursive https://github.com/SHI-Labs/NATTEN.git
cd NATTEN
git checkout v0.21.6
git submodule update --init --recursive
```

### 4. Fix source files — replace C++ `not` keyword with `!`

MSVC does not support the alternative token `not` for `!` in C++. NATTEN uses it extensively. Replace it before building:

```powershell
cd csrc
Get-ChildItem -Recurse -Include *.cu,*.cuh,*.h,*.cpp | ForEach-Object {
    $content = Get-Content $_.FullName -Raw -Encoding UTF8
    $new = $content -creplace '\bnot\b', '!'
    if ($content -ne $new) {
        Set-Content $_.FullName $new -NoNewline -Encoding UTF8
        Write-Host "Fixed: $($_.FullName)"
    }
}
cd ..
```

> This only replaces standalone `not` tokens (not substrings like `not_eq` or `"not found"`). Still, review the changes — any `not` inside string literals that should remain (e.g. error messages) will need to be reverted manually.

### 5. Build

The repo includes `WindowsWhlBuilder.bat` in the root — it handles the build. You need to call it from a **Developer Command Prompt** so MSVC tools are on the PATH.

Open **x64 Native Tools Command Prompt for VS 2022** (from Start Menu), then:

```bat
cd /d C:\path\to\NATTEN
C:\natten-env\Scripts\activate
set NATTEN_CUDA_ARCH=8.6
WindowsWhlBuilder.bat clean
WindowsWhlBuilder.bat build CUDA_ARCH=8.6 WORKERS=8
```

> Set `CUDA_ARCH` to your GPU's compute capability (see table above).
> Set `WORKERS` to the number of **CPU cores** you want to use for parallel compilation (e.g. a 16-core CPU → `WORKERS=16`). More workers = faster build but more RAM usage.

This produces a `.whl` file in the `dist\` folder.

### 6. Install the wheel

```bat
pip install dist\NATTEN-*.whl --force-reinstall --no-deps
```

### 7. Verify

```python
import torch
import natten
from natten import functional as F

q = torch.randn(1, 32, 32, 4, 64, device="cuda", dtype=torch.float32)
k = torch.randn_like(q)
v = torch.randn_like(q)
out = F.na2d(q, k, v, kernel_size=3)
print(f"Output shape: {out.shape}")
```

> Note: NATTEN uses `(B, H, W, Heads, Dim)` ordering, not `(B, C, H, W)`.

---

## Known Issues and Fixes

### 1. `python3X.lib` not found (linker error)

**Symptom:** `LINK : fatal error LNK1181: cannot open input file 'python312.lib'`

**Cause:** Microsoft Store Python does not ship `python3X.lib` (the import library needed for linking C extensions).

**Fix A — Use python.org installer** (recommended). It ships `libs\python3X.lib` next to `python.exe`.

**Fix B — Generate a stub lib** from the DLL:

```bat
set VENV=C:\natten-env
dumpbin /exports "%VENV%\Scripts\python312.dll" > "%TEMP%\python312.exports"
echo EXPORTS > "%TEMP%\python312.def"
for /f "skip=19 tokens=4" %%a in ('type "%TEMP%\python312.exports"') do echo   %%a >> "%TEMP%\python312.def"
if not exist "%VENV%\Scripts\libs" mkdir "%VENV%\Scripts\libs"
lib /def:"%TEMP%\python312.def" /out:"%VENV%\Scripts\libs\python312.lib" /machine:x64
```

> Replace `312` with your Python version (e.g. `311`, `313`). The `libs` directory must exist on the library search path — NATTEN's CMakeLists.txt adds `link_directories("${PY_LIB_DIR}")` for Windows builds.

### 2. `nvToolsExt` not found (CMake error)

**Symptom:** `Could NOT find nvToolsExt` during CMake configure.

**Cause:** PyTorch's bundled `TorchConfig.cmake` references `nvToolsExt` (deprecated in CUDA 12+, removed in CUDA 13).

**Fix:** Comment out `nvToolsExt` references in two files inside your venv:

**File 1:** `%VENV%\Lib\site-packages\torch\share\cmake\Torch\TorchConfig.cmake`
```cmake
# Around line 135, comment out:
# set(CAFFE2_FOUND_nvToolsExt TRUE)
# find_library(CAFFE2_nvToolsExt_LIBRARY NAMES nvToolsExt)
# list(APPEND Caffe2_PUBLIC_LIBRARIES ${CAFFE2_nvToolsExt_LIBRARY})
```

**File 2:** `%VENV%\Lib\site-packages\torch\share\cmake\Caffe2\FindCUDAToolkit.cmake`
```cmake
# Around line 1049, comment out the nvToolsExt block:
# if(NOT CUDA_nvToolsExt_LIBRARY)
#   ...
# endif()
```

### 3. `std: ambiguous symbol` in PyTorch headers

**Symptom:** `error C2872: 'std': ambiguous symbol` in `compiled_autograd.h` or `edge.h`.

**Cause:** PyTorch uses `using namespace torch::autograd` in headers that also reference `std::string`. On MSVC this creates ambiguity between `::std` and the imported namespace.

**Fix:** Edit `%VENV%\Lib\site-packages\torch\include\torch\csrc\dynamo\compiled_autograd.h`.

Find the line (around 1111):
```cpp
#if defined(_WIN32) && (defined(USE_CUDA) || defined(USE_ROCM))
```

Change to:
```cpp
#if defined(_WIN32)
```

This forces the `TORCH_CHECK_NOT_IMPLEMENTED` branch on all Windows builds, which skips the problematic `if constexpr` block entirely.

### 4. C++ `not` keyword not recognized by MSVC

**Symptom:** `error C2065: 'not': undeclared identifier` or similar.

**Cause:** MSVC does not support the alternative token `not` as a keyword unless `<iso646.h>` is included. NATTEN uses `not` freely (Linux/gcc accepts it).

**Fix:** See Step 4 above — bulk replace `not` with `!` in all NATTEN source files.

### 5. `setuptools` rejects `pyproject.toml`

**Symptom:** `Package discovery failed` or `package_dir` errors during build.

**Cause:** Some setuptools versions reject `where = ["src/"]` (with trailing slash) or require specific syntax.

**Fix:** Ensure `pyproject.toml` has `where = ["src"]` (no trailing `/`), or pin `setuptools<69`.

### 6. Linux-only compiler flags on Windows

**Symptom:** `error: unrecognized command line option '-Wconversion'` or similar.

**Cause:** As of NATTEN v0.21.6, the CMakeLists.txt already guards Linux-only flags (`-Wconversion`, `-fno-strict-aliasing`, `-ldl`, `-std=c++17`) with `if(NOT WIN32)`. If you're on an older version, you may need to add these guards yourself in `csrc/CMakeLists.txt`.

---

## Quick Troubleshooting Checklist

| Error | Fix |
|---|---|
| `LNK1181: python3XX.lib` | Generate stub lib or use python.org installer |
| `Could NOT find nvToolsExt` | Comment out nvToolsExt in PyTorch cmake files |
| `C2872: 'std': ambiguous symbol` | Patch `compiled_autograd.h` Win32 guard |
| `C2065: 'not': undeclared identifier` | Replace `not` with `!` in NATTEN sources |
| `unrecognized option '-Wconversion'` | Update NATTEN or guard Linux flags in CMakeLists.txt |
| `Package discovery failed` | Fix `pyproject.toml` or pin `setuptools<69` |
