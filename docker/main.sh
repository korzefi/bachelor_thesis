#!/bin/bash

check_required_bash_version() {
  local required_bash_version="4.5.0"
  if [ "$(printf '%s\n' "$required_bash_version" "${BASH_VERSION}" | sort -V | head -n1)" != "required_bash_version" ]; then
    echo "Update bash version."
    echo "Required bash version is at least $required_bash_version. Current bash version is ${BASH_VERSION}"
    exit 1
  fi
}
check_required_bash_version

BASE_IMAGE_NAME=covid-base:ubuntu
APP_IMAGE_NAME=covid-app:1.0
CONTAINER_NAME=covid_mutations

check_if_project_root_directory() {
  local curr_dir=$(basename "`pwd`")
  if [[ "$curr_dir" != "$ROOT_DIR_NAME" ]]; then
    local script_name=$(basename "$0")
    echo "Run this script from project root directory: sudo bash docker/$script_name"
    exit 1
  fi
}

build_base_image() {
  if [ -z "$(docker images -q "$BASE_IMAGE_NAME" 2> /dev/null)" ]; then
    docker build -f docker/base.Dockerfile -t "$BASE_IMAGE_NAME" .
  fi
}

build_app_image() {
  if [ -z "$(docker images -q "$APP_IMAGE_NAME" 2> /dev/null)" ]; then
    docker build -f docker/Dockerfile -t "$APP_IMAGE_NAME" .
  fi
}

build_images() {
  build_base_image
  build_app_image
}

run_app_container() {
  docker rm -f "$CONTAINER_NAME" || true 2> /dev/null
  docker run \
   -d --network host \
   --name "$CONTAINER_NAME" \
   "$APP_IMAGE_NAME"
}

check_if_project_root_directory
echo "Building images"
build_images
echo "Beginning proccess..."
run_app_container
