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
# Find GPU with third lowest memory usage
GPU_ID_MAIN_CONTAINER=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -n -k2 | awk -F', ' '{print $1}' | head -n 3 | tail -n 1)


## -------------------- CARLA Number of Replicas ------------------
CARLA_SERVER_REPLICAS=${CARLA_SERVER_REPLICAS:-"5"}
CARLA_DEBUG_SERVER_REPLICAS=${CARLA_DEBUG_SERVER_REPLICAS:-"1"}


## -------------------------- CUDA Version ------------------------
CUDA_VERSION=${CUDA_VERSION:-"11.8.0"}


## -------------------------- Write to .env -----------------------
echo "FIXUID=$FIXUID" >> "$SCRIPT_DIR/.env"
echo "FIXGID=$FIXGID" >> "$SCRIPT_DIR/.env"

echo "CARLA_VERSION=$CARLA_VERSION" >> "$SCRIPT_DIR/.env"
echo "CARLA_QUALITY=$CARLA_QUALITY" >> "$SCRIPT_DIR/.env"

echo "GPU_ID_CARLA_MAIN=$GPU_ID_CARLA_MAIN" >> "$SCRIPT_DIR/.env"
echo "GPU_ID_CARLA_DEBUG=$GPU_ID_CARLA_DEBUG" >> "$SCRIPT_DIR/.env"
echo "GPU_ID_MAIN_CONTAINER=$GPU_ID_MAIN_CONTAINER" >> "$SCRIPT_DIR/.env"

echo "CARLA_SERVER_REPLICAS=$CARLA_SERVER_REPLICAS" >> "$SCRIPT_DIR/.env"
echo "CARLA_DEBUG_SERVER_REPLICAS=$CARLA_DEBUG_SERVER_REPLICAS" >> "$SCRIPT_DIR/.env"

echo "CUDA_VERSION=$CUDA_VERSION" >> "$SCRIPT_DIR/.env"