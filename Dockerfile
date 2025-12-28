# Stage 1: Base with all system dependencies
FROM python:3.12-slim AS base

# Install system dependencies including Cairo for CairoSVG
RUN apt-get update && apt-get install -y \
    git \
    curl \
    nodejs \
    npm \
    gosu \
    # Cairo and dependencies for CairoSVG
    libcairo2 \
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf-2.0-dev \
    libffi-dev \
    shared-mime-info \
    # For Pillow
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    # For OpenCV
    libgl1 \
    libglib2.0-0 \
    # Useful tools
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# Install Poetry system-wide
RUN pip install poetry && \
    poetry config virtualenvs.in-project true

# Stage 2: Development image
FROM base AS dev

# Create entrypoint script
RUN printf '#!/bin/bash\n\
set -e\n\
USER_ID=${LOCAL_UID:-1000}\n\
GROUP_ID=${LOCAL_GID:-1000}\n\
USERNAME=dev\n\
getent group $GROUP_ID > /dev/null 2>&1 || groupadd -g $GROUP_ID $USERNAME\n\
id -u $USERNAME > /dev/null 2>&1 || useradd -m -u $USER_ID -g $GROUP_ID -s /bin/bash $USERNAME\n\
chown -R $USERNAME:$GROUP_ID /home/$USERNAME 2>/dev/null || true\n\
exec gosu $USERNAME "$@"\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash", "-c", "poetry config virtualenvs.in-project true && poetry install --no-interaction 2>/dev/null; exec claude --dangerously-skip-permissions"]
