#!/bin/bash

#check_required_bash_version() {
#  local required_bash_version="4.5.0"
#  if [ "$(printf '%s\n' "$required_bash_version" "${BASH_VERSION}" | sort -V | head -n1)" != "required_bash_version" ]; then
#    echo "Update bash version."
#    echo "Required bash version is at least $required_bash_version. Current bash version is ${BASH_VERSION}"
#    exit 1
#  fi
#}
#check_required_bash_version

ROOT_DIR_NAME="bachelor_thesis"
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
  echo "Building base image..."
  build_base_image
  echo "Base image created"

  echo "Building app image..."
  build_app_image
  echo "App image created"
}

run_app_container() {
  docker rm -f "$CONTAINER_NAME" || true 2> /dev/null
  docker run \
   -d --network host \
   --name "$CONTAINER_NAME" \
   "$APP_IMAGE_NAME"
   docker logs --follow
}

copy_result_model() {
  docker cp "$CONTAINER_NAME":data/model ./data/model
}

check_if_project_root_directory
echo "Building images"
build_images
echo "Beginning proccess..."
run_app_container
echo "Process finished."
copy_result_model
echo "Model copied to 'data/model/covid-model.pth'"
