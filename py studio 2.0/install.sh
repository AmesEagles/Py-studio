#!/usr/bin/env bash
set -euo pipefail

# Dependency installer for PyFL Studio on Debian/Ubuntu/Linux Mint.
if ! command -v apt-get >/dev/null 2>&1; then
  echo "This installer currently supports Debian/Ubuntu/Linux Mint (apt-get)." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv python3-pyqt6 \
  python3-numpy python3-scipy python3-soundfile ffmpeg libportaudio2 portaudio19-dev \
  python3-dev build-essential

# Use a project-local virtual environment so we do not modify the OS-managed Python.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv --system-site-packages "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install sounddevice mido python-rtmidi soundfile

echo
echo "Installed. Launch with: $ROOT/.venv/bin/python '$ROOT/pyfl_studio_fixed(1).py'"
echo "If you rename the Python file, update that command accordingly."
