# This derived image rebuilds only the local IPC worker, with inherited dependencies.
ARG C101_BASE_IMAGE
FROM ${C101_BASE_IMAGE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG C101_BASE_IMAGE_ID
LABEL org.neurokinematics.c101.base-image-id="${C101_BASE_IMAGE_ID}"
COPY base-environment-lock.json /opt/c101/runtime/base-environment-lock.json
COPY input-manifest.json /opt/c101/runtime/input-manifest.json
COPY Dockerfile /opt/c101/runtime/Dockerfile
COPY .dockerignore /opt/c101/runtime/.dockerignore
COPY scripts/audit_c101_runtime_build.py /opt/c101/runtime/audit_c101_runtime_build.py
RUN python3 /opt/c101/runtime/audit_c101_runtime_build.py --base-lock /opt/c101/runtime/base-environment-lock.json
COPY src /work/src
COPY scripts /work/scripts
COPY tests /work/tests
COPY ros2_ws/src/c101_moveit_worker /work/ros2_ws/src/c101_moveit_worker
RUN source /opt/ros/jazzy/setup.bash && source /opt/c101/ws/install/setup.bash && \
    cd /opt/c101/ws && \
    colcon build --packages-select c101_moveit_worker --executor sequential \
      --cmake-clean-cache --cmake-clean-first --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF && \
    test -x install/c101_moveit_worker/lib/c101_moveit_worker/c101_moveit_worker && \
    python3 /opt/c101/runtime/audit_c101_runtime_build.py --base-lock /opt/c101/runtime/base-environment-lock.json \
      --input-manifest /opt/c101/runtime/input-manifest.json
WORKDIR /work
