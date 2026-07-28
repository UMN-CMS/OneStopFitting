#!/usr/bin/env bash


container="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:3.11-cuda"

if [[ -n "$SINGULARITY_NAME" ]] || [[ -n "$APPTAINER_NAME" ]]; then
    echo "Inside container: $APPTAINER_NAME"
    if [ -f "$HOME/.bashrc" ]; then
        . "$HOME/.bashrc"
    fi
    echo $PATH

    if ! command -v uv &> /dev/null; then
        echo "Error: 'uv' is not installed or not in PATH."
        echo "Please install 'uv' and try again."
    fi

    echo "Found uv: $(uv --version)"


    SYNC_CMD_EXTRAS=""

    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        uv venv --clear --no-managed-python --relocatable --system-site-packages --link-mode copy
        echo "Syncing dependencies..."
        echo "Running: uv sync --no-managed-python --link-mode copy $SYNC_CMD_EXTRAS"
        uv sync --no-managed-python --link-mode copy $SYNC_CMD_EXTRAS
    fi
    
    exec /bin/bash 
else
    HOST=$(hostname)
    IS_LPC="false"
    if [[ "$HOST" == *"cmslpc"* ]]; then
        echo "Detected LPC host."
        IS_LPC="true"
    fi
    SCRIPT_PATH=$(realpath "$0")
    SCRIPT_PATH=$0
    ARGS=""
    if [[ "$IS_LPC" == "true" ]]; then
        ARGS="-B $HOME,/uscmst1b_scratch,$(realpath "$HOME"),/uscms_data,/cvmfs,/etc/condor/,/usr/local/bin/cmslpc-local-conf.py,$(realpath .):$PWD"
    else
        if [[ -d $(realpath "$HOME/.config") ]]; then 
            ARGS="-B $(realpath .):$PWD,/local/cms/user/bendi083/,$(realpath "$HOME"),/cvmfs,$(realpath "$HOME/.local"),$(realpath "$HOME/.cache"),$(realpath "$HOME/.config")"
        else
            ARGS="-B $(realpath .):$PWD,/local/cms/user/bendi083/,$(realpath "$HOME"),/cvmfs"
        fi
        
    fi

    echo "Entering container..."
    export IS_LPC
    echo $SCRIPT_PATH
    echo apptainer exec $ARGS "$container" /bin/bash  "$SCRIPT_PATH"
    apptainer exec $ARGS "$container" /bin/bash  "$SCRIPT_PATH"
fi 
