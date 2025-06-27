#!/bin/bash

echo "Running Semantic Name Generator Tests..."
echo "========================================"

# Run all tests with coverage
python3 -m pytest tests/ -v --tb=short

# Check if tests passed
if [ $? -eq 0 ]; then
    echo ""
    echo "All tests passed! ✓"
else
    echo ""
    echo "Some tests failed! ✗"
    exit 1
fi