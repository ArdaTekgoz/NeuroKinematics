#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
mkdir -p /opt/c101/external /opt/c101/ws/src /opt/c101/evidence

fetch_source() {
  local name="$1" url="$2" commit="$3"
  git init "/opt/c101/external/$name"
  git -C "/opt/c101/external/$name" remote add origin "$url"
  git -C "/opt/c101/external/$name" fetch --depth 1 origin "$commit"
  git -C "/opt/c101/external/$name" checkout --detach FETCH_HEAD
  test "$(git -C "/opt/c101/external/$name" rev-parse HEAD)" = "$commit"
  ln -s "/opt/c101/external/$name" "/opt/c101/ws/src/$name"
}
fetch_source moveit2 https://github.com/moveit/moveit2.git 093360fef2ea8269c607821b551ffb2d4bc5c9a8
fetch_source trac_ik https://github.com/traclabs/trac_ik.git d8d54abda51f10554501854a744f21fd5b63daf6
fetch_source pick_ik https://github.com/PickNikRobotics/pick_ik.git c476d3aab7879c54f6167ecdf7a1a7fa769de4de
ln -s /work/ros2_ws/src/c101_moveit_worker /opt/c101/ws/src/c101_moveit_worker
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then rosdep init; fi
rosdep update --rosdistro jazzy
cd /opt/c101/ws
mapfile -t selected_paths < <(colcon list --packages-up-to c101_moveit_worker moveit_kinematics trac_ik_kinematics_plugin pick_ik --paths-only)
test "${#selected_paths[@]}" -gt 0
printf '%s\n' "${selected_paths[@]}" > /opt/c101/evidence/selected-source-packages.txt
rosdep keys --from-paths "${selected_paths[@]}" --ignore-src --rosdistro jazzy \
  -t build -t buildtool -t build_export -t buildtool_export -t exec > /opt/c101/evidence/rosdep-keys.txt
packages=()
while IFS= read -r key; do
  [ -n "$key" ] || continue
  resolution="$(rosdep resolve --rosdistro jazzy "$key")"
  printf '%s\n%s\n' "$key" "$resolution" >> /opt/c101/evidence/rosdep-resolutions.txt
  if [ "$(printf '%s\n' "$resolution" | head -n 1)" != '#apt' ]; then
    printf 'Non-apt dependency requires explicit lock: %s\n' "$key" >&2
    exit 2
  fi
  while IFS= read -r package_line; do
    read -ra resolved_packages <<< "$package_line"
    packages+=("${resolved_packages[@]}")
  done < <(printf '%s\n' "$resolution" | tail -n +2)
done < /opt/c101/evidence/rosdep-keys.txt
python3 /work/scripts/c101_install_apt.py --output /opt/c101/evidence/ros-apt-plan.json "${packages[@]}"
colcon build --packages-up-to c101_moveit_worker moveit_kinematics trac_ik_kinematics_plugin pick_ik \
  --parallel-workers 2 --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
dpkg-query -W -f='${binary:Package}\t${Version}\t${Architecture}\n' > /opt/c101/evidence/dpkg-closure.tsv
find /var/cache/apt/archives -maxdepth 1 -name '*.deb' -type f -exec sha256sum {} + \
  | sort > /opt/c101/evidence/deb-cache-sha256.txt
c++ --version > /opt/c101/evidence/compiler.txt
ldd --version > /opt/c101/evidence/libc.txt
test -x /opt/c101/ws/install/c101_moveit_worker/lib/c101_moveit_worker/c101_moveit_worker
