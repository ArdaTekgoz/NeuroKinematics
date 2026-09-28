ARG ROS_BASE_IMAGE
FROM ${ROS_BASE_IMAGE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=2 RCUTILS_LOGGING_USE_STDOUT=0
COPY scripts/c101_install_apt.py /opt/c101_install_apt.py
# Retain downloaded debs inside the image as the exact dependency snapshot.
RUN rm -f /etc/apt/apt.conf.d/docker-clean && apt-get update && \
    python3 /opt/c101_install_apt.py --output /opt/c101/evidence/bootstrap-apt-plan.json \
      build-essential cmake git curl ca-certificates python3-colcon-common-extensions python3-rosdep nlohmann-json3-dev
RUN curl -fL --retry 3 https://github.com/prefix-dev/pixi/releases/download/v0.81.0/pixi-x86_64-unknown-linux-musl \
      -o /usr/local/bin/pixi && chmod +x /usr/local/bin/pixi && \
    test "$(pixi --version)" = 'pixi 0.81.0' && \
    sha256sum /usr/local/bin/pixi > /opt/c101/evidence/pixi-binary-sha256.txt
WORKDIR /work
COPY pixi.toml pixi.lock pyproject.toml README.md /work/
COPY src /work/src
COPY assets /work/assets
COPY config /work/config
COPY experiments /work/experiments
COPY scripts /work/scripts
COPY ros2_ws/src /work/ros2_ws/src
COPY tests/c1_01 /work/tests/c1_01
COPY data/generated/F0-05/acceptance/run-a/query-list.jsonl /work/data/generated/F0-05/acceptance/run-a/query-list.jsonl
RUN pixi install --locked
# Shell files may originate in a Windows CRLF checkout.
RUN sed -i 's/\r$//' scripts/c101_build_linux.sh && bash scripts/c101_build_linux.sh
RUN mkdir -p /opt/c101/asset_prefix/share/ament_index/resource_index/packages && \
    touch /opt/c101/asset_prefix/share/ament_index/resource_index/packages/kuka_agilus_support && \
    ln -s /work/assets/robots/robot_a/source_snapshot/kuka_agilus_support /opt/c101/asset_prefix/share/kuka_agilus_support
RUN printf '%s\n' '#!/usr/bin/env bash' 'set -eo pipefail' \
    'source /opt/ros/jazzy/setup.bash' 'source /opt/c101/ws/install/setup.bash' \
    'export AMENT_PREFIX_PATH="/opt/c101/asset_prefix:${AMENT_PREFIX_PATH}"' \
    'cd /work' 'exec pixi run --locked "$@"' > /opt/c101/entrypoint.sh && chmod +x /opt/c101/entrypoint.sh
ENTRYPOINT ["/opt/c101/entrypoint.sh"]
CMD ["python", "-m", "neurokinematics.core.cli", "--help"]
