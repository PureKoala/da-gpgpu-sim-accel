#!/bin/bash

# This script orchestrates the entire process of testing the custom FMR instruction.
# 1. It navigates to the test directory.
# 2. It invokes the Makefile to:
#    a. Compile the CUDA source (`test_fmr_magic.cu`) to an intermediate PTX file.
#    b. Run the Python patcher script (`ptx_patcher.py`) to replace the "magic"
#       function call with our custom `ld.sample.fmr.f16` instruction.
#    c. Compile the host-side code for verification purposes (optional).
# 3. It runs the GPGPU-Sim simulator with the final, patched PTX file.

# Exit on first error
set -e

# Get the directory of this script to robustly find the test directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TEST_DIR="$SCRIPT_DIR/test"

echo "--- Navigating to test directory: $TEST_DIR ---"
cd "$TEST_DIR"

# Run the 'run' target in the Makefile.
# This will build the necessary files and then execute the simulation.
echo "--- Invoking Makefile to build and run simulation ---"
make run

echo "--- Simulation run finished ---"
echo "--- To clean up generated files, run: 'make clean' in the '$TEST_DIR' directory ---"
