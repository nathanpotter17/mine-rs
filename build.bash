#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

BUILD_MODE=""

shaders=(
    "basic.vert:basic.vert.spv"
    "basic.frag:basic.frag.spv"
    "crosshair.vert:crosshair.vert.spv"
    "crosshair.frag:crosshair.frag.spv"
    "sky.vert:sky.vert.spv"
    "sky.frag:sky.frag.spv"
)

compile_shaders() {
    echo -e "${CYAN}Compiling shaders...${NC}"
    mkdir -p shaders/compiled
    for shader in "${shaders[@]}"; do
        IFS=':' read -r src dst <<< "$shader"
        echo -e "  Compiling $src..."
        if ! glslc "shaders/$src" -o "shaders/compiled/$dst" --target-env=vulkan1.1 --target-spv=spv1.3; then
            echo -e "${RED}✗ Shader compilation failed for $src${NC}"
            exit 1
        fi
        echo -e "  ${GREEN}✓${NC} $src"
    done
}

build_engine() {
    compile_shaders
    
    echo -e "${CYAN}Building engine ($BUILD_MODE)...${NC}"
    
    if [ "$BUILD_MODE" = "release" ]; then
        cargo build --release
    else
        cargo build
    fi
    
    echo -e "${GREEN}✓ Engine build complete${NC}"
}

run_engine() {
    local binary="./target/${BUILD_MODE:-release}/simmer"
    
    if [ ! -f "$binary" ]; then
        if [ -f "./target/release/simmer" ]; then
            binary="./target/release/simmer"
        elif [ -f "./target/debug/simmer" ]; then
            binary="./target/debug/simmer"
        else
            echo -e "${RED}Engine not built! Use option 1 to build.${NC}"
            exit 1
        fi
    fi
    
    echo -e "${CYAN}Running engine...${NC}"
    $binary
}

echo -e "\n${CYAN}===== VOXEL WORLD BUILD SYSTEM =====${NC}\n"
echo -e "${GREEN}Options:${NC}"
echo "  1) Build & Run (Debug)"
echo "  2) Build & Run (Release)"
echo "  3) Compile Shaders Only"
echo "  4) Run (no build)"
read -p "Selection [1-4]: " choice

case "$choice" in
    1) BUILD_MODE="debug"; build_engine; run_engine ;;
    2) BUILD_MODE="release"; build_engine; run_engine ;;
    3) compile_shaders ;;
    4) run_engine ;;
    *) echo -e "${RED}Invalid selection${NC}"; exit 1 ;;
esac

echo -e "\n${GREEN}Done!${NC}"