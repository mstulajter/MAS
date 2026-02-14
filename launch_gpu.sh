#!/bin/bash

# ---- Debug info ----
echo "[$(hostname)] PID $$"
echo "OMPI_COMM_WORLD_RANK=$OMPI_COMM_WORLD_RANK"
echo "OMPI_COMM_WORLD_LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK"
echo "Working directory: $(pwd)"

# ---- Set GPU per local MPI rank ----
if [[ -n "$OMPI_COMM_WORLD_LOCAL_RANK" ]]; then
    export CUDA_VISIBLE_DEVICES="$OMPI_COMM_WORLD_LOCAL_RANK"
else
    # Non-MPI run fallback
    export CUDA_VISIBLE_DEVICES=0
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ---- Validate input file if provided ----
if [[ -n "$3" ]]; then
    if [[ ! -f "$3" ]]; then
        echo "ERROR: Input file '$3' not found in $(pwd)"
        exit 1
    fi
fi

# ---- Execute program ----
exec "$@"
