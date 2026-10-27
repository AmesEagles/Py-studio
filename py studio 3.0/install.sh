#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# PyFL Studio - dependency installer
#
#   ./install.sh               install everything into ./.venv and create ./run.sh
#   ./install.sh --no-venv     install into the Python you already use instead
#   ./install.sh --no-system   skip OS packages (PortAudio, libsndfile, ffmpeg...)
#
# Works on Linux (apt / dnf / pacman / zypper), macOS (Homebrew) and Windows
# (Git Bash / MSYS2 - the pip wheels already bundle PortAudio and libsndfile).
# ----------------------------------------------------------------------------
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

USE_VENV=1
USE_SYSTEM=1
for arg in "$@"; do
  case "$arg" in
    --no-venv)   USE_VENV=0 ;;
    --no-system) USE_SYSTEM=0 ;;
    -h|--help)   sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)"; exit 1 ;;
  esac
done

if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'; else B=""; G=""; Y=""; R=""; N=""; fi
say()  { echo "${B}==>${N} $*"; }
ok()   { echo "${G}  OK${N}  $*"; }
warn() { echo "${Y}WARN${N}  $*"; }
die()  { echo "${R}FAIL${N}  $*"; exit 1; }

# ---- platform -------------------------------------------------------------
case "$(uname -s 2>/dev/null || echo unknown)" in
  Linux*)               PLAT=linux ;;
  Darwin*)              PLAT=mac ;;
  MINGW*|MSYS*|CYGWIN*) PLAT=windows ;;
  *)                    PLAT=other ;;
esac
SUDO=""
if [ "$(id -u 2>/dev/null || echo 0)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi
FAILED_PKGS=()

install_each() {  # install_each <cmd...> -- <pkg...>   (one by one, so a missing name never aborts the rest)
  local cmd=() pkgs=() seen=0
  for a in "$@"; do
    if [ "$a" = "--" ]; then seen=1; elif [ $seen -eq 0 ]; then cmd+=("$a"); else pkgs+=("$a"); fi
  done
  for p in "${pkgs[@]}"; do
    "${cmd[@]}" "$p" >/dev/null 2>&1 && echo "      + $p" || { FAILED_PKGS+=("$p"); echo "      - $p (not installed)"; }
  done
}

install_system_packages() {
  if [ "$PLAT" = linux ]; then
    if command -v apt-get >/dev/null 2>&1; then
      say "Installing system packages with apt (you may be asked for your password)"
      $SUDO apt-get update -y >/dev/null 2>&1 || warn "apt-get update failed"
      install_each $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -- \
        python3 python3-venv python3-pip python3-dev build-essential pkg-config \
        libportaudio2 portaudio19-dev libsndfile1 ffmpeg libasound2-dev libjack-jackd2-dev \
        libxcb-cursor0 libxkbcommon-x11-0 libxkbcommon0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 \
        libegl1 libgl1 libpulse0 libdbus-1-3 libfontconfig1
    elif command -v dnf >/dev/null 2>&1; then
      say "Installing system packages with dnf"
      install_each $SUDO dnf install -y -- \
        python3 python3-pip python3-devel gcc gcc-c++ make portaudio portaudio-devel libsndfile ffmpeg-free \
        alsa-lib-devel jack-audio-connection-kit-devel xcb-util-cursor libxkbcommon-x11 mesa-libEGL mesa-libGL pulseaudio-libs
    elif command -v pacman >/dev/null 2>&1; then
      say "Installing system packages with pacman"
      install_each $SUDO pacman -S --noconfirm --needed -- \
        python python-pip base-devel portaudio libsndfile ffmpeg alsa-lib jack2 xcb-util-cursor libxkbcommon-x11 mesa
    elif command -v zypper >/dev/null 2>&1; then
      say "Installing system packages with zypper"
      install_each $SUDO zypper --non-interactive install -- \
        python3 python3-pip python3-devel gcc gcc-c++ make portaudio-devel libsndfile1 ffmpeg alsa-devel \
        libjack-devel libxcb-cursor0 libxkbcommon-x11-0 Mesa-libEGL1 Mesa-libGL1
    else
      warn "Unknown Linux package manager - install PortAudio, libsndfile and ffmpeg yourself."
    fi
  elif [ "$PLAT" = mac ]; then
    if command -v brew >/dev/null 2>&1; then
      say "Installing system packages with Homebrew"
      install_each brew install -- portaudio libsndfile ffmpeg
      command -v python3 >/dev/null 2>&1 || install_each brew install -- python
    else
      warn "Homebrew not found (https://brew.sh). Install it, then run: brew install portaudio libsndfile ffmpeg"
    fi
  else
    say "Skipping OS packages on this platform (pip wheels bundle what is needed; ffmpeg is optional)"
  fi
  if [ ${#FAILED_PKGS[@]} -gt 0 ]; then
    warn "Some system packages could not be installed: ${FAILED_PKGS[*]}"
    warn "Usually harmless - the script keeps going and checks everything at the end."
  fi
}

[ "$USE_SYSTEM" -eq 1 ] && install_system_packages

# ---- python ---------------------------------------------------------------
PYBIN=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
    PYBIN="$c"; break
  fi
done
[ -n "$PYBIN" ] || die "Python 3.9 or newer was not found. Install it from https://www.python.org/downloads/ and run this script again."
ok "Using $($PYBIN --version 2>&1) ($(command -v "$PYBIN"))"

if [ "$USE_VENV" -eq 1 ]; then
  say "Creating virtual environment in ./.venv"
  if [ ! -d .venv ]; then
    "$PYBIN" -m venv .venv || die "Could not create a virtual environment. On Debian/Ubuntu run: sudo apt install python3-venv"
  fi
  if [ -x .venv/bin/python ]; then VPY=".venv/bin/python"; else VPY=".venv/Scripts/python.exe"; fi
  [ -e "$VPY" ] || die "Virtual environment looks broken - delete the .venv folder and run again."
else
  VPY="$PYBIN"
fi
if [ "$USE_VENV" -eq 1 ]; then
  VPY_ABS="$(cd "$(dirname "$VPY")" && pwd)/$(basename "$VPY")"
else
  VPY_ABS="$(command -v "$PYBIN")"
fi

say "Installing Python packages"
"$VPY" -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || warn "Could not upgrade pip (continuing)"
REQUIRED=(numpy scipy PyQt6 sounddevice soundfile mido miniaudio)
if ! "$VPY" -m pip install "${REQUIRED[@]}"; then
  warn "Bulk install failed - retrying one package at a time"
  for p in "${REQUIRED[@]}"; do
    "$VPY" -m pip install "$p" || warn "Could not install $p"
  done
fi
# MIDI keyboard support: python-rtmidi needs a compiler/ALSA headers on some systems, so it is optional.
"$VPY" -m pip install python-rtmidi >/dev/null 2>&1 \
  && ok "python-rtmidi (live MIDI keyboard) installed" \
  || warn "python-rtmidi could not be built - MIDI keyboard input is disabled (everything else works)"

# ---- verify ---------------------------------------------------------------
say "Checking the installation"
if "$VPY" - <<'PY'
import importlib, shutil, sys
BAD = []
def check(mod, label, needed=True):
    try:
        m = importlib.import_module(mod)
        print(f"  OK    {label:<14} {getattr(m, '__version__', '')}")
        return m
    except Exception as e:
        print(f"  {'FAIL' if needed else 'WARN'}  {label:<14} {type(e).__name__}: {e}")
        if needed:
            BAD.append(label)
        return None
check("numpy", "numpy"); check("scipy", "scipy"); check("PyQt6.QtWidgets", "PyQt6")
sd = check("sounddevice", "sounddevice")
sf = check("soundfile", "soundfile")
check("miniaudio", "miniaudio", False); check("mido", "mido"); check("rtmidi", "python-rtmidi", False)
if sf is not None:
    fmts = sf.available_formats()
    print("  OK    audio formats  " + ", ".join(k for k in ("WAV", "FLAC", "OGG", "MP3") if k in fmts))
    if "MP3" not in fmts:
        print("  note  MP3 will be decoded by miniaudio / ffmpeg instead")
print("  " + ("OK    " if shutil.which("ffmpeg") else "WARN  ") + "ffmpeg         " + (shutil.which("ffmpeg") or "not found (optional fallback decoder)"))
if sd is not None:
    try:
        devs = sd.query_devices()
        ins = [d for d in devs if d["max_input_channels"] > 0]
        outs = [d for d in devs if d["max_output_channels"] > 0]
        print(f"  OK    audio devices  {len(outs)} output, {len(ins)} input (microphone needed for 'Hum notes')")
    except Exception as e:
        print("  WARN  audio devices  could not be queried:", e)
sys.exit(1 if BAD else 0)
PY
then
  INSTALL_OK=1
else
  INSTALL_OK=0
fi

# ---- launcher -------------------------------------------------------------
cat > run.sh <<EOF
#!/usr/bin/env bash
cd "\$(dirname "\${BASH_SOURCE[0]}")" || exit 1
exec "$VPY_ABS" pyfl_studio.py "\$@"
EOF
chmod +x run.sh 2>/dev/null

echo
if [ "$INSTALL_OK" -eq 1 ]; then
  say "Done!  Start PyFL Studio with:   ${B}./run.sh${N}"
else
  warn "Some required packages are missing (see the FAIL lines above). Fix them (internet connection? missing"
  warn "system library?) and run ./install.sh again. ./run.sh was still created."
fi
if [ "$PLAT" = mac ]; then
  echo "     macOS: the first time you use 'Hum notes', allow microphone access for your terminal when asked."
fi
if [ "$PLAT" = linux ]; then
  echo "     Linux: if the window does not open, make sure you are in a desktop session (X11 or Wayland)."
fi
