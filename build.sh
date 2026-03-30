#!/bin/bash

# Exit immediately if any command fails
set -e

echo "=== Starting LAMBDA Build ==="

# Create the build directory if it doesn't already exist
mkdir -p build

# Navigate into the build directory
cd build

# Run CMake to generate the Makefiles
echo "--- Configuring CMake ---"
cmake ..

# Compile the code
echo "--- Compiling Project ---"
make -j4

echo "--- Installing LAMBDA ---"
cd ..
mkdir -p bin
mv build/lambda bin/lambda

echo ""
echo "=== Build Successful! ==="