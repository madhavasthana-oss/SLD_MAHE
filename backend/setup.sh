#!/usr/bin/env bash
# =============================================================================
# setup.sh — ASL backend setup for Windows (Git Bash)
# Installs Python 3.12, creates venv, installs deps, runs the server
# Run from your backend folder: bash setup.sh
# =============================================================================

set -e  # stop on first error

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC}   $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
die()     { echo -e "${RED}[ERR]${NC}  $1"; exit 1; }

# ── Config ────────────────────────────────────────────────────────────────────
PY_VERSION="3.12.9"
PY_INSTALLER="python-${PY_VERSION}-amd64.exe"
PY_URL="https://www.python.org/ftp/python/${PY_VERSION}/${PY_INSTALLER}"
VENV_DIR="venv312"
BACKEND_DIR="$(pwd)"

# ── Step 0: Make sure we're in the right folder ───────────────────────────────
info "Working directory: $BACKEND_DIR"
if [[ ! -f "main.py" ]]; then
    die "main.py not found. Run this script from your backend folder."
fi

# ── Step 1: Find or install Python 3.12 ──────────────────────────────────────
info "Looking for Python 3.12..."

PY312=""

# Common install locations for Python 3.12 on Windows
CANDIDATES=(
    "/c/Users/$USERNAME/AppData/Local/Programs/Python/Python312/python.exe"
    "/c/Program Files/Python312/python.exe"
    "/c/Python312/python.exe"
    "/c/Users/$USERNAME/AppData/Local/Programs/Python/Python312/python3.12.exe"
)

for candidate in "${CANDIDATES[@]}"; do
    if [[ -f "$candidate" ]]; then
        ver=$("$candidate" --version 2>&1)
        if [[ "$ver" == *"3.12"* ]]; then
            PY312="$candidate"
            success "Found Python 3.12 at: $PY312"
            break
        fi
    fi
done

# Also try PATH
if [[ -z "$PY312" ]]; then
    for cmd in python3.12 python3 python; do
        if command -v "$cmd" &>/dev/null; then
            ver=$("$cmd" --version 2>&1)
            if [[ "$ver" == *"3.12"* ]]; then
                PY312=$(command -v "$cmd")
                success "Found Python 3.12 in PATH: $PY312"
                break
            fi
        fi
    done
fi

# Not found — download and install it
if [[ -z "$PY312" ]]; then
    warn "Python 3.12 not found. Downloading installer..."

    TEMP_DIR=$(mktemp -d)
    INSTALLER_PATH="$TEMP_DIR/$PY_INSTALLER"

    curl -L --progress-bar "$PY_URL" -o "$INSTALLER_PATH" \
        || die "Failed to download Python 3.12. Check your internet connection."

    success "Downloaded $PY_INSTALLER"
    info "Launching installer — when it opens:"
    info "  1. Check 'Add python.exe to PATH'"
    info "  2. Click 'Install Now'"
    info "  3. Come back here when done"
    echo ""

    # Launch installer and wait for it to finish
    "$INSTALLER_PATH" /quiet InstallAllUsers=0 PrependPath=1 \
        DefaultAllUsersTargetDir="" \
        TargetDir="$USERPROFILE\\AppData\\Local\\Programs\\Python\\Python312" \
        || {
            warn "Silent install may have failed. Trying interactive install..."
            cmd.exe //c "start /wait $INSTALLER_PATH"
        }

    # Re-scan after install
    sleep 2
    FRESH_PATH="/c/Users/$USERNAME/AppData/Local/Programs/Python/Python312/python.exe"
    if [[ -f "$FRESH_PATH" ]]; then
        PY312="$FRESH_PATH"
        success "Python 3.12 installed at: $PY312"
    else
        die "Python 3.12 install finished but can't find python.exe.\nClose this terminal, reopen Git Bash, and run: bash setup.sh"
    fi
fi

# Confirm version
PYVER=$("$PY312" --version 2>&1)
success "Using: $PYVER ($PY312)"

# ── Step 2: Create virtual environment ───────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    warn "venv '$VENV_DIR' already exists — skipping creation."
else
    info "Creating virtual environment in ./$VENV_DIR ..."
    "$PY312" -m venv "$VENV_DIR" || die "Failed to create venv."
    success "venv created."
fi

# ── Step 3: Activate venv ─────────────────────────────────────────────────────
info "Activating venv..."
# shellcheck disable=SC1091
source "$VENV_DIR/Scripts/activate" || die "Could not activate venv. Try: source $VENV_DIR/Scripts/activate"
success "venv active. Python: $(python --version)"

# ── Step 4: Upgrade pip ───────────────────────────────────────────────────────
info "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# ── Step 5: Install dependencies ─────────────────────────────────────────────
info "Installing dependencies (this may take a few minutes)..."

pip install \
    fastapi \
    "uvicorn[standard]" \
    mediapipe \
    torch \
    torchvision \
    opencv-python \
    ultralytics \
    --quiet \
    && success "All dependencies installed." \
    || die "pip install failed. See errors above."

# ── Step 6: Verify mediapipe actually loads ───────────────────────────────────
info "Verifying mediapipe Tasks API..."
python -c "
import mediapipe.tasks as tasks
lm = tasks.vision.HandLandmarker
print('mediapipe HandLandmarker: OK')
" || die "mediapipe import failed even on 3.12. Check the error above."

success "mediapipe verified."

# ── Step 7: Check for hand_landmarker.task ────────────────────────────────────
if [[ ! -f "hand_landmarker.task" ]]; then
    info "Downloading hand_landmarker.task model (~8MB)..."
    curl -L --progress-bar \
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
        -o hand_landmarker.task \
        && success "hand_landmarker.task downloaded." \
        || warn "Could not download hand_landmarker.task. Download it manually and place in this folder."
else
    success "hand_landmarker.task already present."
fi

# ── Step 8: Launch server ─────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  All good. Starting ASL backend server...  ${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
info "API: http://127.0.0.1:8000"
info "Docs: http://127.0.0.1:8000/docs"
info "Press Ctrl+C to stop."
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload