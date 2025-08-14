#!/bin/bash

# Docker deployment script for SARS-CoV-2 Mutation Prediction Pipeline
# Supports modular pipeline execution with volume mapping and argument passing
# 
# Usage: 
#   bash docker/main.sh [--prepare|--cluster|--train|--test|--full]
#   bash docker/main.sh --config path/to/config.yaml --prepare
#   bash docker/main.sh --build-only  # Just build images without running

set -e  # Exit on any error

# Configuration
ROOT_DIR_NAME="bachelor_thesis"
BASE_IMAGE_NAME="covid-base:ubuntu"
APP_IMAGE_NAME="covid-app:1.0"
CONTAINER_NAME="covid_mutations"
DEFAULT_CONFIG="configs/sars_cov_2_default.yaml"

# Parse command line arguments
PIPELINE_ARGS=""
BUILD_ONLY=false
CONFIG_PATH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --build-only)
      BUILD_ONLY=true
      shift
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --prepare|--cluster|--train|--test|--full)
      PIPELINE_ARGS="$PIPELINE_ARGS $1"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS] [PIPELINE_COMMANDS]"
      echo ""
      echo "OPTIONS:"
      echo "  --build-only        Build images without running pipeline"
      echo "  --config PATH       Use specific config file (default: $DEFAULT_CONFIG)"
      echo "  --help, -h          Show this help message"
      echo ""
      echo "PIPELINE_COMMANDS:"
      echo "  --prepare           Run data preparation step"
      echo "  --cluster           Run clustering step"
      echo "  --train             Run training step"
      echo "  --test              Run testing step"
      echo "  --full              Run complete pipeline (prepare -> cluster -> train -> test)"
      echo ""
      echo "EXAMPLES:"
      echo "  $0 --prepare"
      echo "  $0 --config custom.yaml --train"
      echo "  $0 --full"
      echo "  $0 --build-only"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Default to full pipeline if no specific commands given
if [[ -z "$PIPELINE_ARGS" && "$BUILD_ONLY" == false ]]; then
  PIPELINE_ARGS="--full"
fi

# Set config path
if [[ -n "$CONFIG_PATH" ]]; then
  CONFIG_ARG="--config $CONFIG_PATH"
else
  CONFIG_ARG="--config $DEFAULT_CONFIG"
fi

check_if_project_root_directory() {
  local curr_dir=$(basename "`pwd`")
  if [[ "$curr_dir" != "$ROOT_DIR_NAME" ]]; then
    local script_name=$(basename "$0")
    echo "ERROR: Run this script from project root directory"
    echo "Usage: bash docker/$script_name [OPTIONS]"
    exit 1
  fi
}

check_docker_availability() {
  if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH"
    exit 1
  fi
  
  if ! docker info &> /dev/null; then
    echo "ERROR: Docker daemon is not running"
    exit 1
  fi
}

build_base_image() {
  echo "Building base image: $BASE_IMAGE_NAME"
  if [ -z "$(docker images -q "$BASE_IMAGE_NAME" 2> /dev/null)" ]; then
    docker build -f docker/base.Dockerfile -t "$BASE_IMAGE_NAME" .
    echo "✓ Base image built successfully"
  else
    echo "✓ Base image already exists"
  fi
}

build_app_image() {
  echo "Building application image: $APP_IMAGE_NAME"
  if [ -z "$(docker images -q "$APP_IMAGE_NAME" 2> /dev/null)" ]; then
    docker build -f docker/Dockerfile -t "$APP_IMAGE_NAME" .
    echo "✓ Application image built successfully"
  else
    echo "✓ Application image already exists"
  fi
}

build_images() {
  echo "===========================================" 
  echo "Building Docker Images"
  echo "==========================================="
  build_base_image
  echo ""
  build_app_image
  echo ""
  echo "✓ All images built successfully"
}

create_volume_directories() {
  # Ensure host directories exist for volume mapping
  echo "Creating necessary directories for volume mapping..."
  mkdir -p data/input data/processed models results
  echo "✓ Directories created"
}

run_pipeline_step() {
  local step_args="$1"
  
  echo "===========================================" 
  echo "Running Pipeline: $step_args"
  echo "==========================================="
  
  # Remove existing container if it exists
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  
  # Run container with volume mapping and argument passing
  echo "Starting container with volume mapping..."
  docker run \
    --name "$CONTAINER_NAME" \
    --rm \
    -v "$(pwd)/data:/bachelor_thesis/data" \
    -v "$(pwd)/models:/bachelor_thesis/models" \
    -v "$(pwd)/results:/bachelor_thesis/results" \
    -v "$(pwd)/configs:/bachelor_thesis/configs" \
    "$APP_IMAGE_NAME" \
    python3 main.py $CONFIG_ARG $step_args
    
  echo "✓ Pipeline step completed successfully"
}

copy_results() {
  echo "===========================================" 
  echo "Results Available"
  echo "==========================================="
  echo "Models saved in: ./models/"
  echo "Results saved in: ./results/"
  echo "Processed data in: ./data/processed/"
  
  # List generated files
  if [[ -d "models" && "$(ls -A models 2>/dev/null)" ]]; then
    echo ""
    echo "Generated models:"
    ls -la models/ | grep -E '\.(pth|json)$' || echo "  No model files found"
  fi
  
  if [[ -d "results" && "$(ls -A results 2>/dev/null)" ]]; then
    echo ""
    echo "Generated results:"
    ls -la results/ | grep -E '\.(json|png|csv)$' || echo "  No result files found"
  fi
}

main() {
  echo "==========================================="
  echo "SARS-CoV-2 Mutation Prediction Pipeline"
  echo "Docker Deployment Script"
  echo "==========================================="
  
  # Validation checks
  check_if_project_root_directory
  check_docker_availability
  
  # Build images
  build_images
  
  # Exit if build-only mode
  if [[ "$BUILD_ONLY" == true ]]; then
    echo "✓ Build completed. Exiting as requested (--build-only)"
    exit 0
  fi
  
  # Prepare host environment
  create_volume_directories
  
  # Execute pipeline
  if [[ "$PIPELINE_ARGS" == *"--full"* ]]; then
    echo "Running complete pipeline..."
    run_pipeline_step "--prepare"
    echo ""
    run_pipeline_step "--cluster" 
    echo ""
    run_pipeline_step "--train"
    echo ""
    run_pipeline_step "--test"
  else
    run_pipeline_step "$PIPELINE_ARGS"
  fi
  
  # Show results
  echo ""
  copy_results
  
  echo ""
  echo "✓ Pipeline execution completed successfully!"
}

# Run main function
main "$@"
