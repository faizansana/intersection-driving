SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# If .env exists, ask user if they want to overwrite
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "Config file already exists. Do you want to overwrite it? [y/N]"
    read -r answer
    if [ "$answer" != "${answer#[Yy]}" ] ;then
        rm "$SCRIPT_DIR/.env"
    else
        exit 1
    fi
fi

## -------------------------- User ID -----------------------------

FIXUID=$(id -u) 
FIXGID=$(id -g) 

## -------------------------- CARLA Version -----------------------

CARLA_VERSION=${CARLA_VERSION:-"0.9.10.1"}
CARLA_QUALITY=${CARLA_QUALITY:-"Low"}

## -------------------------- Determine GPU with Lowest Memory Usage ------------------------

# Find the GPU with the lowest memory usage
GPU_ID_CARLA_MAIN=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -n -k2 | awk -F', ' '{print $1}' | head -n 1)
# Find GPU with second lowest memory usage
GPU_ID_CARLA_DEBUG=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -n -k2 | awk -F', ' '{print $1}' | head -n 2 | tail -n 1)

## -------------------------- CUDA Version ------------------------
CUDA_VERSION=${CUDA_VERSION:-"11.8.0"}


echo "FIXUID=$FIXUID" >> "$SCRIPT_DIR/.env"
echo "FIXGID=$FIXGID" >> "$SCRIPT_DIR/.env"

echo "CARLA_VERSION=$CARLA_VERSION" >> "$SCRIPT_DIR/.env"
echo "CARLA_QUALITY=$CARLA_QUALITY" >> "$SCRIPT_DIR/.env"

echo "GPU_ID_CARLA_MAIN=$GPU_ID_CARLA_MAIN" >> "$SCRIPT_DIR/.env"
echo "GPU_ID_CARLA_DEBUG=$GPU_ID_CARLA_DEBUG" >> "$SCRIPT_DIR/.env"

echo "CUDA_VERSION=$CUDA_VERSION" >> "$SCRIPT_DIR/.env"