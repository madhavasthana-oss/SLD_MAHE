#!/bin/bash

# ── Colors ────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
WHITE='\033[1;37m'
DIM='\033[2m'
RESET='\033[0m'

# ── Progress printer ──────────────────────────────────────────
# Usage: step <n> <color> <message>
# n = 1..5, bracket is always 13 chars wide: [ * * * * * ]
step() {
  local n=$1
  local color=$2
  local msg=$3
  local stars=""
  for (( i=1; i<=5; i++ )); do
    if (( i <= n )); then
      stars+="*"
    else
      stars+=" "
    fi
    if (( i < 5 )); then stars+=" "; fi
  done
  echo -e "${color}[ ${stars} ]  ${msg}${RESET}"
}

# ── Header ────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}  <<<<<<<<<<<< ASL DETECTION >>>>>>>>>>>>  ${RESET}"
echo -e "${DIM}  -------- ResNet50 x MediaPipe --------  ${RESET}"
echo ""

# ── Step 1 — locate backend ───────────────────────────────────
step 1 $CYAN "locating backend..."
cd backend || {
  echo -e "${RED}  [ERROR] backend/ folder not found${RESET}"
  exit 1
}
step 1 $GREEN "backend/ found"
echo ""

# ── Step 2 — venv check / create ─────────────────────────────
step 2 $CYAN "checking venv312..."

if [ ! -d "venv312" ]; then
  step 2 $YELLOW "venv312 not found — creating with Python 3.12..."

  if command -v py &> /dev/null; then
    py -3.12 -m venv venv312
  elif command -v python3.12 &> /dev/null; then
    python3.12 -m venv venv312
  else
    echo -e "${RED}  [ERROR] Python 3.12 not found. Install it and retry.${RESET}"
    exit 1
  fi

  step 2 $GREEN "venv312 created"
  echo ""

  # ── Step 3 — activate ─────────────────────────────────────
  step 3 $CYAN "activating virtual environment..."
  source venv312/Scripts/activate
  step 3 $GREEN "virtual environment active"
  echo ""

  # ── Step 4 — install dependencies ─────────────────────────
  step 4 $YELLOW "installing dependencies (this will take a while)..."
  echo ""

  pip install --upgrade pip --quiet

  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
  pip install \
    mediapipe \
    fastapi \
    "uvicorn[standard]" \
    wandb \
    matplotlib \
    seaborn \
    numpy \
    tqdm \
    opencv-python \
    --quiet

  echo ""
  step 4 $GREEN "all dependencies installed"

else
  step 2 $GREEN "venv312 found"
  echo ""

  # ── Step 3 — activate (existing venv) ─────────────────────
  step 3 $CYAN "activating virtual environment..."
  source venv312/Scripts/activate
  step 3 $GREEN "virtual environment active"
fi

echo ""

# ── Step 5 — launch ───────────────────────────────────────────
step 5 $WHITE "launching backend — http://127.0.0.1:8000"
echo ""

uvicorn main:app --reload

echo ""
echo -e "${DIM}  xx  ASL-Detector closed  xx${RESET}"
echo ""