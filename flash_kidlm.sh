#!/usr/bin/env bash
# Flash the KidLM-5M quantized INT8 model weights directly to XIAO ESP32-S3 Flash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-/dev/ttyACM0}"
BIN_FILE="${2:-$SCRIPT_DIR/kidlm_checkpoints/kidlm_5m_q8.bin}"
OFFSET="0x190000"

if [ ! -f "$BIN_FILE" ]; then
    echo "Error: Model binary $BIN_FILE does not exist! Train and export the model first."
    exit 1
fi

echo "=========================================================="
echo " Flashing KidLM-5M Model to Seeed Studio XIAO ESP32-S3"
echo " Target Port:   $PORT"
echo " Flash Offset:  $OFFSET (Partition Size: 0x660000 = 6.375 MB)"
echo " Model Binary:  $BIN_FILE"
echo "=========================================================="

if [ ! -e "$PORT" ]; then
    echo "Warning: Port $PORT not found! Plug in the board and check 'ls -l /dev/ttyACM*'."
    echo "If board is busy or in sleep mode, hold BOOT (B), press RESET (R), release BOOT."
    exit 1
fi

sudo chmod 666 "$PORT" 2>/dev/null || true

/home/xhroot/snap/antigravity-cli/common/local/bin/uv run --with esptool esptool \
    --chip esp32s3 \
    -p "$PORT" \
    -b 921600 \
    --before default-reset \
    --after hard-reset \
    write-flash -z \
    "$OFFSET" "$BIN_FILE"

echo ""
echo "=========================================================="
echo " KidLM-5M flashed successfully to partition at $OFFSET!"
echo "=========================================================="
