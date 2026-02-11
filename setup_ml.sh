#!/bin/bash
set -e

echo "Setting up Arcane development environment..."

# Check for mamba, then micrcomamba, then conda
if command -v mamba &> /dev/null; then
    CMD="mamba"
elif command -v micromamba &> /dev/null; then
    CMD="micromamba"
elif command -v conda &> /dev/null; then
    CMD="conda"
else
    echo "Conda/Mamba not found. Please install Mambaforge or Miniconda."
    exit 1
fi

echo "Using $CMD for environment setup."

# Check if environment exists
if $CMD env list | grep -q "arcane"; then
    echo "Updating environment 'arcane'..."
    $CMD env update -n arcane -f environment.yml --prune
else
    echo "Creating environment 'arcane'..."
    $CMD env create -f environment.yml
fi

echo ""
echo "Setup complete!"
echo "To activate the environment, run:"
echo "conda activate arcane"
echo ""
echo "To run the app:"
echo "python src/main.py"
