#!/usr/bin/env bash

CONTAINER="dodo-sim2seal-jazzy"
IMAGE="dodo-sim2seal:jazzy"
REPO_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

SOURCE_CMD='
[ -f /workspace/jazzy_ws/install/setup.bash ] && source /workspace/jazzy_ws/install/setup.bash
[ -f /repo/jazzy_ws/install/setup.bash ] && source /repo/jazzy_ws/install/setup.bash
export PYTHONPATH=/usr/local/lib/python3.12/dist-packages:/usr/lib/python3/dist-packages:$PYTHONPATH
echo "[dodo] ROS2 ready — container: '"$CONTAINER"'"
cd /repo/jazzy_ws
exec bash
'

if [ "$1" = "stop" ]; then
    echo "[dodo] Stopping container..."
    docker stop "$CONTAINER" && docker rm "$CONTAINER"
    exit 0
fi

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "[dodo] Starting container..."
    docker run -d \
        --name "$CONTAINER" \
        --network host \
        -e ROS_DOMAIN_ID=0 \
        -e FASTRTPS_DEFAULT_PROFILES_FILE=/repo/jazzy_ws/fastdds.xml \
        -v "$REPO_DIR:/repo" \
        -w /repo \
        "$IMAGE" \
        sleep infinity
    echo "[dodo] Container started."
    echo "[dodo] Installing compatibility dependencies..."
    docker exec "$CONTAINER" pip3 install --break-system-packages --quiet setuptools==68.1.2
    docker exec "$CONTAINER" pip3 install --break-system-packages --quiet psutil
    echo "[dodo] Dependencies ready."
fi

echo "[dodo] Attaching..."
docker exec -it "$CONTAINER" bash -c "$SOURCE_CMD"
