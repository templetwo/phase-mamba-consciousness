#!/bin/bash
# Monitor K-SSM v3 training on Mac Studio from local machine
# Usage: ./monitor_remote.sh

REMOTE_HOST="tony_studio@192.168.1.195"
REMOTE_LOG="~/liminal-k-ssm/results/kssm_v3/training.log"
LOCAL_CACHE="/tmp/kssm_v3_training.log"

echo "=========================================="
echo "K-SSM v3 Remote Training Monitor"
echo "=========================================="
echo "Remote: $REMOTE_HOST"
echo "Log: $REMOTE_LOG"
echo ""

# Check if remote is accessible
if ! ssh -q "$REMOTE_HOST" exit; then
    echo "❌ Error: Cannot connect to $REMOTE_HOST"
    echo "   Check SSH connection and credentials"
    exit 1
fi

# Check if remote log exists
if ! ssh "$REMOTE_HOST" "test -f $REMOTE_LOG"; then
    echo "⚠️  Warning: Log file not found on remote"
    echo "   Expected: $REMOTE_LOG"
    echo ""
    echo "Available logs:"
    ssh "$REMOTE_HOST" "find ~/liminal-k-ssm -name 'training.log' -type f 2>/dev/null"
    exit 1
fi

echo "✓ Connected to Mac Studio"
echo "✓ Log file found"
echo ""
echo "Starting live monitor (CTRL+C to exit)..."
echo ""

# Method 1: Direct tail (simpler, no local caching)
ssh "$REMOTE_HOST" "tail -f $REMOTE_LOG" | while read -r line; do
    echo "$line"

    # Parse and colorize key metrics
    if [[ "$line" =~ ^[[:space:]]*([0-9]+)[[:space:]]*\| ]]; then
        # This is a metrics line - could add live coloring here
        :
    fi
done

# Alternative Method 2: Rsync + local monitor (more features)
# Uncomment to use Python monitor with remote sync:
#
# echo "Syncing log to local cache: $LOCAL_CACHE"
# while true; do
#     rsync -az --quiet "$REMOTE_HOST:$REMOTE_LOG" "$LOCAL_CACHE" 2>/dev/null
#     sleep 2
# done &
# SYNC_PID=$!
#
# # Wait for initial sync
# sleep 3
#
# # Run local Python monitor
# cd "$(dirname "$0")/.."
# python3 kssm/monitor_training.py --log-file "$LOCAL_CACHE"
#
# # Cleanup on exit
# kill $SYNC_PID 2>/dev/null
