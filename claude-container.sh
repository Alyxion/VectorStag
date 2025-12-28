#!/bin/bash

# claude-container.sh - Run Claude Code in a sandboxed Docker environment

set -e
cd "$(dirname "$0")"

# Export UID/GID for docker-compose
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)

# Build if needed and run
case "${1:-claude}" in
    shell)
        echo "Starting shell..."
        docker compose --profile shell run --rm shell
        ;;
    build)
        echo "Building image..."
        docker compose build
        ;;
    down)
        docker compose down
        ;;
    *)
        echo "Starting Claude Code..."
        docker compose run --rm claude
        ;;
esac
