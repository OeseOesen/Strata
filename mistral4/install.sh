#!/usr/bin/env bash
# mistral4/install.sh - builds strata-m4 (Mistral Small 4 on one V100 + a two-socket host) and its helpers on Ubuntu.
#
#   ./mistral4/install.sh [--prefix DIR] [--arch 70] [--no-llamacpp] [--systemd] [--model FILE.gguf]
#                         [--mmproj FILE.gguf] [--tokenizer tekken.json|hf-repo] [--cuda-home DIR]
#
# --cuda-home: a CUDA 12.x toolkit outside /usr/local, e.g. a conda env (~/cuda12, ~/miniforge3/envs/cuda12) made with
#   `conda create -p ~/cuda12 -c nvidia/label/cuda-12.6.3 cuda-toolkit=12.6` - no root needed.  Found by itself there too.
#
# What it does (nothing outside --prefix, except apt packages and, with --systemd, one unit file):
#   1. checks the GPU and the driver, and finds a CUDA 12.x toolkit (CUDA 13 has no Volta); installs cuda-toolkit-12-9
#      from NVIDIA's apt repository when none is there and you agree
#   2. builds strata-m4 from this checkout (sm_70)
#   3. builds llama.cpp at a pinned commit (sm_70): the reference for speed and perplexity, and libmtmd for images
#   4. builds m4-vision against it, makes a Python venv with mistral-common for m4serve.py
#   5. writes $PREFIX/m4.env (paths and options) and, with --systemd, /etc/systemd/system/strata-m4.service
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PREFIX="${HOME}/strata-m4"
ARCH=70
LLAMA_COMMIT=1094646c1fb706b0bc7cf3569c707a173032a01d
WITH_LLAMA=1
SYSTEMD=0
MODEL=""
MMPROJ=""
TOKENIZER="mistralai/Mistral-Small-4-119B-2603"
JOBS="$(nproc)"
CUDA_HOME_OPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --no-llamacpp) WITH_LLAMA=0; shift ;;
    --systemd) SYSTEMD=1; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --mmproj) MMPROJ="$2"; shift 2 ;;
    --tokenizer) TOKENIZER="$2"; shift 2 ;;
    --cuda-home) CUDA_HOME_OPT="$2"; shift 2 ;;
    -j) JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
mkdir -p "$PREFIX/bin" "$PREFIX/src" "$PREFIX/logs"

say "GPU and driver"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: the NVIDIA driver is missing"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,pci.bus_id --format=csv,noheader
DRV_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
[[ "$DRV_MAJOR" -ge 525 ]] || die "driver $DRV_MAJOR is too old for CUDA 12 (need >= 525)"
BUS="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr 'A-Z' 'a-z' | sed 's/^0000//; s/^\(....\):/0000:/')"
for d in /sys/bus/pci/devices/*; do
  if [[ "$(basename "$d")" == *"${BUS: -7}" ]]; then echo "GPU NUMA node: $(cat "$d/numa_node")"; fi
done
command -v lscpu >/dev/null && lscpu | grep -E "^(Model name|Socket|NUMA node[0-9]* CPU|Thread)" || true

say "build tools"
NEED=()
for p in build-essential git cmake python3-venv python3-dev numactl pkg-config; do
  dpkg -s "$p" >/dev/null 2>&1 || NEED+=("$p")
done
if [[ ${#NEED[@]} -gt 0 ]]; then
  echo "missing packages: ${NEED[*]}"
  if sudo -n true 2>/dev/null; then sudo apt-get update -q && sudo apt-get install -y -q "${NEED[@]}"
  else echo "(no passwordless sudo: continuing; the conda env and pip supply what is missing where they can)"; fi
fi
command -v git >/dev/null || die "git is needed"
command -v python3 >/dev/null || die "python3 is needed"
say "CUDA 12.x toolkit"
NVCC=""
cuda12() { local n="$1/bin/nvcc"; [[ -x "$n" ]] && "$n" --version | grep -q "release 12\." ; }
for c in "$CUDA_HOME_OPT" "${CONDA_PREFIX:-}" "$HOME/cuda12" "$HOME/miniforge3/envs/cuda12" "$HOME/miniconda3/envs/cuda12" \
         /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12 /usr/local/cuda; do
  [[ -n "$c" ]] && cuda12 "$c" && { NVCC="$c/bin/nvcc"; break; }
done
if [[ -z "$NVCC" ]] && command -v nvcc >/dev/null && nvcc --version | grep -q "release 12\."; then NVCC="$(command -v nvcc)"; fi
if [[ -z "$NVCC" ]]; then
  echo "No CUDA 12 toolkit found (CUDA 13 cannot build for the V100)."
  echo "Root-free: conda create -p ~/cuda12 -c nvidia/label/cuda-12.6.3 cuda-toolkit=12.6 gxx_linux-64=12, then --cuda-home ~/cuda12"
  read -r -p "Or install cuda-toolkit-12-9 from NVIDIA's apt repository now (sudo)? [y/N] " ok
  [[ "$ok" == "y" || "$ok" == "Y" ]] || die "install a CUDA 12.x toolkit and run this again"
  . /etc/os-release
  REPO_ID="ubuntu${VERSION_ID//./}"
  tmp="$(mktemp -d)"
  wget -q -O "$tmp/keyring.deb" "https://developer.download.nvidia.com/compute/cuda/repos/${REPO_ID}/x86_64/cuda-keyring_1.1-1_all.deb"
  sudo dpkg -i "$tmp/keyring.deb"
  sudo apt-get update -q
  sudo apt-get install -y -q cuda-toolkit-12-9   # the toolkit only: the installed driver stays
  NVCC=/usr/local/cuda-12.9/bin/nvcc
fi
CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
echo "nvcc: $NVCC ($("$NVCC" --version | grep release))"
export PATH="$CUDA_HOME/bin:$PATH"
CUDA_FLAGS=(-DCMAKE_CUDA_COMPILER="$NVCC" -DCUDAToolkit_ROOT="$CUDA_HOME")
# a conda toolkit: its libraries (and the driver stubs for linking) are under lib/, and its host compiler matches it
if [[ -d "$CUDA_HOME/conda-meta" ]]; then
  # headers: conda keeps cuda_runtime.h under targets/x86_64-linux/include, which Strata's plain C++ files need too.
  # No LIBRARY_PATH: the env's lib/ holds conda's libm.so linker script pointing at /lib64 (CentOS layout), which
  # breaks the system linker; CMake finds cudart and cuBLAS through CUDAToolkit_ROOT by full path
  unset LIBRARY_PATH
  export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include:${CPATH:-}"
  HOSTCXX=""
  for g in "$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++" "$CUDA_HOME/bin/g++"; do [[ -x "$g" ]] && { HOSTCXX="$g"; break; }; done
  SYSGCC="$(g++ -dumpversion 2>/dev/null | cut -d. -f1)"
  if [[ -n "$SYSGCC" && "$SYSGCC" -le 13 ]]; then HOSTCXX="$(command -v g++)"; fi   # CUDA 12.6 takes gcc <= 13
  [[ -n "$HOSTCXX" ]] && CUDA_FLAGS+=(-DCMAKE_CUDA_HOST_COMPILER="$HOSTCXX")
  CUDA_FLAGS+=(-DCMAKE_BUILD_RPATH="$CUDA_HOME/lib" -DCMAKE_INSTALL_RPATH="$CUDA_HOME/lib")
  # link line: nvcc reports the env's lib/ as an implicit link directory, and there -lm finds conda's libm.so script
  # (/lib64/libm.so.6, a CentOS path).  -L options are searched in command-line order, so the system's library
  # directory goes first; the static CUDA runtime (cudart_static, cudadevrt) sits under targets/x86_64-linux/lib.
  MA="$(gcc -print-multiarch 2>/dev/null || echo x86_64-linux-gnu)"
  LINKF="-L/usr/lib/$MA -L/lib/$MA -L$CUDA_HOME/targets/x86_64-linux/lib -L$CUDA_HOME/lib"
  CUDA_FLAGS+=(-DCMAKE_EXE_LINKER_FLAGS="$LINKF" -DCMAKE_SHARED_LINKER_FLAGS="$LINKF" -DCMAKE_MODULE_LINKER_FLAGS="$LINKF")
  [[ -n "$HOSTCXX" ]] || die "system g++ $SYSGCC is too new for CUDA 12 (<= 13) and the conda env has no g++ (conda install -p $CUDA_HOME gxx_linux-64=12)"
  echo "conda toolkit: CUDA host compiler $HOSTCXX ($("$HOSTCXX" -dumpversion)), libraries from $CUDA_HOME/lib"
fi

say "cmake / python"
mkvenv() {
  [[ -x "$PREFIX/venv/bin/pip" ]] && return 0
  rm -rf "$PREFIX/venv"
  python3 -m venv "$PREFIX/venv" >/dev/null 2>&1 && [[ -x "$PREFIX/venv/bin/pip" ]] && return 0
  rm -rf "$PREFIX/venv"
  # no ensurepip (python3-venv missing, no sudo): the conda env's python, or a venv without pip + get-pip
  if [[ -x "$CUDA_HOME/bin/python3" ]]; then "$CUDA_HOME/bin/python3" -m venv "$PREFIX/venv" && return 0; rm -rf "$PREFIX/venv"; fi
  python3 -m venv --without-pip "$PREFIX/venv" || return 1
  curl -sS https://bootstrap.pypa.io/get-pip.py | "$PREFIX/venv/bin/python" - -q || return 1
  [[ -x "$PREFIX/venv/bin/pip" ]]
}
mkvenv || die "no Python venv possible (install python3-venv, or put python into the conda env)"
CMAKE=""
for c in "$CUDA_HOME/bin/cmake" "$(command -v cmake 2>/dev/null)" "$PREFIX/venv/bin/cmake"; do
  [[ -n "$c" && -x "$c" ]] || continue
  v="$("$c" --version | head -1 | awk '{print $3}')"
  if [[ "$(printf '%s\n3.24\n' "$v" | sort -V | head -1)" == "3.24" ]]; then CMAKE="$c"; break; fi
done
if [[ -z "$CMAKE" ]]; then
  "$PREFIX/venv/bin/pip" install -q "cmake>=3.28" "ninja"
  CMAKE="$PREFIX/venv/bin/cmake"
fi
echo "cmake: $CMAKE ($("$CMAKE" --version | head -1))"

say "strata-m4 (sm_$ARCH)"
BUILD="$PREFIX/build-m4"
"$CMAKE" -S "$REPO/engine" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  "${CUDA_FLAGS[@]}" -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_INKLING=OFF -DSTRATA_BUILD_INKLING2=OFF -DSTRATA_BUILD_M4=ON \
  > "$PREFIX/logs/cmake-m4.log" 2>&1 || { tail -40 "$PREFIX/logs/cmake-m4.log"; die "cmake failed (log: $PREFIX/logs/cmake-m4.log)"; }
"$CMAKE" --build "$BUILD" --target strata-m4 -j "$JOBS" 2>&1 | tee "$PREFIX/logs/build-m4.log" | grep -E "error|warning: unused|Built target strata-m4" || true
[[ -x "$BUILD/strata-m4" ]] || die "strata-m4 did not build (log: $PREFIX/logs/build-m4.log)"
cp "$BUILD/strata-m4" "$PREFIX/bin/"

if [[ $WITH_LLAMA -eq 1 ]]; then
  say "llama.cpp $LLAMA_COMMIT (reference + libmtmd)"
  LC="$PREFIX/src/llama.cpp"
  if [[ ! -d "$LC/.git" ]]; then git clone -q https://github.com/ggml-org/llama.cpp "$LC"; fi
  git -C "$LC" fetch -q origin && git -C "$LC" checkout -q "$LLAMA_COMMIT"
  "$CMAKE" -S "$LC" -B "$LC/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON "${CUDA_FLAGS[@]}" \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=ON \
    > "$PREFIX/logs/cmake-llama.log" 2>&1 || { tail -40 "$PREFIX/logs/cmake-llama.log"; die "llama.cpp cmake failed"; }
  "$CMAKE" --build "$LC/build" -j "$JOBS" --target llama-server llama-bench llama-perplexity llama-tokenize llama-cli mtmd \
    > "$PREFIX/logs/build-llama.log" 2>&1 || { tail -40 "$PREFIX/logs/build-llama.log"; die "llama.cpp build failed"; }
  say "m4-vision"
  "$CMAKE" -S "$HERE/vision" -B "$PREFIX/build-vision" -DLLAMA_CPP_DIR="$LC" -DLLAMA_CPP_BUILD="$LC/build" \
    -DCMAKE_BUILD_TYPE=Release > "$PREFIX/logs/cmake-vision.log" 2>&1 || die "m4-vision cmake failed"
  "$CMAKE" --build "$PREFIX/build-vision" -j "$JOBS" > "$PREFIX/logs/build-vision.log" 2>&1 || die "m4-vision build failed"
  cp "$PREFIX/build-vision/m4-vision" "$PREFIX/bin/"
fi

say "Python environment for m4serve"
"$PREFIX/venv/bin/pip" install -q --upgrade pip
"$PREFIX/venv/bin/pip" install -q "mistral-common[image]>=1.13" "huggingface_hub"
# strata monitor's terminal UI (Strata's own pin), so the monitor runs from this venv
if [[ -f "$REPO/terminal/requirements-monitor.txt" ]]; then
  "$PREFIX/venv/bin/pip" install -q -r "$REPO/terminal/requirements-monitor.txt" || echo "note: textual for strata monitor not installed"
fi
cp "$HERE/m4serve.py" "$HERE/m4cli.py" "$HERE/bench.sh" "$HERE/parity.sh" "$HERE/sanity.sh" "$HERE/speed.sh" "$HERE/longctx.sh" "$HERE/start-m4.sh" "$HERE/monitor-m4.sh" "$HERE/parallel_check.py" "$PREFIX/"
chmod +x "$PREFIX/bench.sh" "$PREFIX/parity.sh" "$PREFIX/sanity.sh" "$PREFIX/speed.sh" "$PREFIX/longctx.sh" "$PREFIX/start-m4.sh" "$PREFIX/monitor-m4.sh"
# the `strata-m4` command (launch, serve, stop, status, monitor, logs, config, doctor, bench ...) on the PATH
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/strata-m4" <<EOS
#!/usr/bin/env bash
export M4_HOME="$PREFIX" M4_SRC="\${M4_SRC:-$REPO}"
exec "$PREFIX/venv/bin/python" "$PREFIX/m4cli.py" "\$@"
EOS
chmod +x "$HOME/.local/bin/strata-m4"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo "note: ~/.local/bin is not on PATH yet (log in again, or: export PATH=\"\$HOME/.local/bin:\$PATH\")";; esac

say "configuration"
ENVF="$PREFIX/m4.env"
if [[ ! -f "$ENVF" ]]; then
  cat > "$ENVF" <<EOF
# strata-m4 settings (read by strata-m4.service and the scripts)
M4_MODEL=${MODEL:-/path/to/Mistral-Small-4-119B-2603-Q4_K_M.gguf}
M4_MMPROJ=${MMPROJ}
M4_TOKENIZER=${TOKENIZER}
# engine options: --ctx (tokens), --kv int8|f16, --spec (n-gram drafts), --numa tiered|split|off,
# --ram-gpu-node / --ram-other-node (GiB per expert arena), --vram-reserve (GiB)
M4_ENGINE_ARGS="--ctx 65536 --kv int8 --spec 6 --numa split"
M4_HOST=127.0.0.1
M4_PORT=8095
# conversations at once (1-8): they share the KV cache of --ctx, each takes pages as it grows (strata-m4 launch --parallel N)
# M4_PARALLEL=1
# M4_API_KEY=change-me
M4_LLAMA=$PREFIX/src/llama.cpp/build/bin
EOF
fi
echo "settings: $ENVF"

if [[ $SYSTEMD -eq 1 ]]; then
  say "systemd unit"
  sudo tee /etc/systemd/system/strata-m4.service >/dev/null <<EOF
[Unit]
Description=strata-m4 (Mistral Small 4, OpenAI API)
After=network.target

[Service]
User=$USER
EnvironmentFile=$ENVF
ExecStart=/bin/bash -c 'exec $PREFIX/venv/bin/python $PREFIX/m4serve.py --engine $PREFIX/bin/strata-m4 --model "\$M4_MODEL" --tokenizer "\$M4_TOKENIZER" \${M4_MMPROJ:+--mmproj "\$M4_MMPROJ" --vision $PREFIX/bin/m4-vision} --engine-args "\$M4_ENGINE_ARGS" --host "\$M4_HOST" --port "\$M4_PORT" \${M4_PARALLEL:+--parallel \$M4_PARALLEL}'
Restart=on-failure
RestartSec=10
LimitMEMLOCK=infinity
TimeoutStartSec=900

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  echo "enable with: sudo systemctl enable --now strata-m4"
fi

say "done"
cat <<EOF
Binaries: $PREFIX/bin (strata-m4, m4-vision), llama.cpp: $PREFIX/src/llama.cpp/build/bin
Next:
  1. set M4_MODEL (and M4_MMPROJ for images) in $ENVF
  2. correctness against llama.cpp:   $PREFIX/parity.sh
  3. speed against llama.cpp:         $PREFIX/bench.sh
  4. serve:  $PREFIX/venv/bin/python $PREFIX/m4serve.py --engine $PREFIX/bin/strata-m4 --model \$M4_MODEL --tokenizer \$M4_TOKENIZER
EOF
