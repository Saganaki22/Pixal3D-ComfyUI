from __future__ import annotations

import argparse
import importlib
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"

REQUIRED_GROUPS = (
    ("flex_gemm", ("flex_gemm_ap", "flex_gemm")),
    ("cumesh", ("cumesh_vb", "cumesh")),
    ("o_voxel", ("o_voxel_vb_ap", "o_voxel")),
    ("drtk", ("drtk",)),
)
ATTENTION_GROUPS = (
    ("FlashAttention 2", ("flash_attn",)),
    ("FlashAttention 3", ("flash_attn_interface",)),
)
OPTIONAL_MODULES = ("triton", "nvdiffrast.torch", "nvdiffrec_render", "natten")
KNOWN_CUDA_PACKAGES = "flex_gemm, cumesh, o_voxel, and drtk"

# Known-good Windows wheel URLs. These are exact-stack installs, not guesses.
# requirements.txt installs plain natten, but strict NAF still needs a matching
# libnatten build. No universal Windows libnatten wheel exists across stacks.
KNOWN_CUDA_WHEELS: dict[tuple[str, str, str, str], list[str]] = {
    ("windows", "cp312", "torch2.10", "cu130"): [
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%2Bcu130torch2.10-cp312-cp312-win_amd64.whl",
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%2Bcu130torch2.10-cp312-cp312-win_amd64.whl",
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel_vb_ap-latest/o_voxel_vb_ap-0.0.1%2Bcu130torch2.10-cp312-cp312-win_amd64.whl",
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%2Bcu130torch2.10-cp312-cp312-win_amd64.whl",
    ],
    ("windows", "cp312", "torch2.9", "cu130"): [
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%2Bcu130torch2.9-cp312-cp312-win_amd64.whl",
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%2Bcu130torch2.9-cp312-cp312-win_amd64.whl",
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel_vb_ap-latest/o_voxel_vb_ap-0.0.1%2Bcu130torch2.9-cp312-cp312-win_amd64.whl",
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%2Bcu130torch2.9-cp312-cp312-win_amd64.whl",
    ],
    ("windows", "cp312", "torch2.9", "cu128"): [
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%2Bcu128torch2.9-cp312-cp312-win_amd64.whl",
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%2Bcu128torch2.9-cp312-cp312-win_amd64.whl",
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel_vb_ap-latest/o_voxel_vb_ap-0.0.1%2Bcu128torch2.9-cp312-cp312-win_amd64.whl",
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%2Bcu128torch2.9-cp312-cp312-win_amd64.whl",
    ],
    ("windows", "cp312", "torch2.8", "cu128"): [
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%2Bcu128torch2.8-cp312-cp312-win_amd64.whl",
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%2Bcu128torch2.8-cp312-cp312-win_amd64.whl",
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel_vb_ap-latest/o_voxel_vb_ap-0.0.1%2Bcu128torch2.8-cp312-cp312-win_amd64.whl",
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%2Bcu128torch2.8-cp312-cp312-win_amd64.whl",
    ],
}

KNOWN_NATTEN_PACKAGES: dict[tuple[str, str, str], str] = {
    ("linux", "torch2.11", "cu130"): "natten==0.21.6+torch2110cu130",
    ("linux", "torch2.11", "cu128"): "natten==0.21.6+torch2110cu128",
    ("linux", "torch2.11", "cu126"): "natten==0.21.6+torch2110cu126",
    ("linux", "torch2.10", "cu130"): "natten==0.21.6+torch2100cu130",
    ("linux", "torch2.10", "cu128"): "natten==0.21.6+torch2100cu128",
    ("linux", "torch2.10", "cu126"): "natten==0.21.6+torch2100cu126",
}


@dataclass(frozen=True)
class RuntimeInfo:
    os_key: str
    python_tag: str
    torch_tag: str | None
    cuda_tag: str | None
    torch_version: str | None
    cuda_version: str | None
    gpu_name: str | None
    gpu_capability: str | None

    @property
    def wheel_key(self) -> tuple[str, str, str, str] | None:
        if self.torch_tag is None or self.cuda_tag is None:
            return None
        return self.os_key, self.python_tag, self.torch_tag, self.cuda_tag


def run(cmd: list[str], *, dry_run: bool = False) -> int:
    print("+ " + " ".join(cmd))
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=str(ROOT))
    return int(completed.returncode)


def pip_install(args: list[str], *, dry_run: bool = False) -> int:
    return run([sys.executable, "-m", "pip", "install", *args], dry_run=dry_run)


def os_key() -> str:
    system = platform.system().lower()
    if system.startswith("win"):
        return "windows"
    if system == "linux":
        return "linux"
    if system == "darwin":
        return "macos"
    return system or "unknown"


def detect_runtime() -> RuntimeInfo:
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_version = None
    cuda_version = None
    torch_tag = None
    cuda_tag = None
    gpu_name = None
    gpu_capability = None

    try:
        import torch

        torch_version = str(getattr(torch, "__version__", ""))
        cuda_version = getattr(torch.version, "cuda", None)
        version_core = torch_version.split("+", 1)[0]
        parts = version_core.split(".")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            torch_tag = f"torch{parts[0]}.{parts[1]}"
        if cuda_version:
            cuda_tag = "cu" + "".join(str(cuda_version).split(".")[:2])
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            gpu_capability = f"{major}.{minor}"
    except Exception as exc:
        torch_version = f"not importable ({type(exc).__name__}: {exc})"

    return RuntimeInfo(
        os_key=os_key(),
        python_tag=py_tag,
        torch_tag=torch_tag,
        cuda_tag=cuda_tag,
        torch_version=torch_version,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_capability=gpu_capability,
    )


def import_status(names: tuple[str, ...]) -> tuple[bool, str]:
    messages = []
    for name in names:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "")
            return True, f"{name} OK" + (f" ({version})" if version else "")
        except Exception as exc:
            messages.append(f"{name}: {type(exc).__name__}: {exc}")
    return False, "; ".join(messages)


def print_runtime_report(info: RuntimeInfo) -> None:
    print("")
    print("Pixal3D-ComfyUI install/runtime check")
    print(f"- Python executable: {sys.executable}")
    print(f"- Node path: {ROOT}")
    print(f"- OS: {info.os_key}")
    print(f"- Python tag: {info.python_tag}")
    print(f"- PyTorch: {info.torch_version}")
    print(f"- torch.version.cuda: {info.cuda_version}")
    print(f"- GPU: {info.gpu_name or 'not detected'}")
    print(f"- GPU capability: {info.gpu_capability or 'not detected'}")
    print(f"- Wheel key: {info.wheel_key or 'not available'}")

    print("")
    print("Required CUDA/Pixal3D modules:")
    missing_required = False
    for label, names in REQUIRED_GROUPS:
        ok, detail = import_status(names)
        missing_required = missing_required or not ok
        print(f"- {label}: {'OK' if ok else 'MISSING'} ({detail})")

    print("")
    print("Attention modules, need at least one:")
    attention_ok = False
    for label, names in ATTENTION_GROUPS:
        ok, detail = import_status(names)
        attention_ok = attention_ok or ok
        print(f"- {label}: {'OK' if ok else 'MISSING'} ({detail})")

    print("")
    print("Optional modules:")
    for name in OPTIONAL_MODULES:
        ok, detail = import_status((name,))
        if name == "natten" and ok:
            try:
                import natten

                detail += f", HAS_LIBNATTEN={getattr(natten, 'HAS_LIBNATTEN', 'unknown')}"
            except Exception:
                pass
        print(f"- {name}: {'OK' if ok else 'not found'} ({detail})")

    print("")
    if attention_ok:
        print("Attention requirement: OK")
    else:
        print("Attention requirement: missing flash_attn or flash_attn_interface")
    if missing_required:
        print("")
        print("Required Pixal3D CUDA modules are missing.")
        print(f"--install-known-cuda installs: {KNOWN_CUDA_PACKAGES}.")
        print("FlashAttention 2 or 3 is a prerequisite and must already match this environment.")
        print("This is opt-in because the wheels must exactly match Python, PyTorch, CUDA, and OS.")
        print("If this detected stack is in the bundled wheel map, rerun with --install-known-cuda or set PIXAL3D_INSTALL_KNOWN_CUDA=1.")
        print("Otherwise install matching wheels from docs/windows_wheels.md.")


def install_requirements(*, dry_run: bool = False) -> None:
    if not REQUIREMENTS.exists():
        raise FileNotFoundError(f"Missing {REQUIREMENTS}")
    code = pip_install(["-r", str(REQUIREMENTS)], dry_run=dry_run)
    if code != 0:
        raise SystemExit(code)


def install_known_cuda(info: RuntimeInfo, *, dry_run: bool = False) -> None:
    key = info.wheel_key
    if key is None or key not in KNOWN_CUDA_WHEELS:
        print("")
        print("No exact known Pixal3D CUDA extension wheel set is bundled for this stack.")
        print(f"Detected key: {key}")
        print("See docs/windows_wheels.md and requirements-cuda-manual.txt.")
        return

    print("")
    print("Installing exact known Pixal3D CUDA extension wheels for:")
    print(key)
    print(f"Package groups: {KNOWN_CUDA_PACKAGES}")
    print("FlashAttention is treated as a prerequisite and is not installed by this option.")
    print("Strict NAF still needs a real libnatten build; plain natten is not enough.")
    code = pip_install(["--no-deps", *KNOWN_CUDA_WHEELS[key]], dry_run=dry_run)
    if code != 0:
        raise SystemExit(code)


def install_known_natten(info: RuntimeInfo, *, dry_run: bool = False) -> None:
    if info.os_key == "windows":
        print("")
        print("Skipping automatic NATTEN on native Windows.")
        print("Upstream NATTEN points Windows users to MSVC source builds; official wheel commands are the easy path for Linux/WSL.")
        print("Use docs/troubleshooting.md if you want to attempt a Windows source build.")
        return

    if info.torch_tag is None or info.cuda_tag is None:
        print("")
        print("Cannot choose a NATTEN wheel because PyTorch/CUDA was not detected.")
        return

    key = (info.os_key, info.torch_tag, info.cuda_tag)
    package = KNOWN_NATTEN_PACKAGES.get(key)
    if package is None:
        print("")
        print("No known official NATTEN+libnatten package for this stack.")
        print(f"Detected key: {key}")
        print("See https://natten.org/install/")
        return

    print("")
    print("Installing official NATTEN+libnatten package for:")
    print(key)
    code = pip_install(["--no-deps", package, "-f", "https://whl.natten.org"], dry_run=dry_run)
    if code != 0:
        raise SystemExit(code)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pixal3D-ComfyUI portable/standalone installer")
    parser.add_argument("--check", action="store_true", help="Only print the runtime/import report")
    parser.add_argument("--skip-requirements", action="store_true", help="Do not install requirements.txt")
    parser.add_argument("--install-known-cuda", action="store_true", help="Install bundled exact-match Pixal3D CUDA extension wheels when available")
    parser.add_argument("--install-natten", action="store_true", help="Install official NATTEN+libnatten package when available for Linux/WSL")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args(argv)

    install_cuda = args.install_known_cuda or os.environ.get("PIXAL3D_INSTALL_KNOWN_CUDA") == "1"
    install_natten = args.install_natten or os.environ.get("PIXAL3D_INSTALL_NATTEN") == "1"

    if not args.check and not args.skip_requirements:
        install_requirements(dry_run=args.dry_run)

    info = detect_runtime()
    if not args.check and install_cuda:
        install_known_cuda(info, dry_run=args.dry_run)
        info = detect_runtime()
    if not args.check and install_natten:
        install_known_natten(info, dry_run=args.dry_run)
        info = detect_runtime()

    print_runtime_report(info)

    print("")
    print("Next steps:")
    print("- Restart ComfyUI after installing or changing wheels.")
    print("- Run the Pixal3D Environment Check node.")
    print("- Keep naf_mode=fallback_if_missing unless natten.HAS_LIBNATTEN is True.")
    print("- Do not let pip change Torch unless you are intentionally rebuilding the ComfyUI stack.")


if __name__ == "__main__":
    main()
