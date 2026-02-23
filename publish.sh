#!/bin/bash
set -e

BUILD_FIRST=true
TEST_PYPI=false

usage() {
    echo "Usage: $0 [--test|--prod]"
    echo "  --test    Build and publish to test.pypi.org"
    echo "  --prod    Build and publish to production PyPI (default)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_PYPI=true
            shift
            ;;
        --prod)
            TEST_PYPI=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/

# Build the package
echo "Building package..."
python -m build

# Verify build
echo "Running twine check..."
twine check dist/*

# Publish
if [ "$TEST_PYPI" = true ]; then
    echo "Publishing to test.pypi.org..."
    twine upload --repository testpypi dist/*
    echo "✅ Published to test.pypi.org"
    echo "   Test install: pip install --index-url https://test.pypi.org/simple/ syrin"
else
    echo "Publishing to production PyPI..."
    twine upload dist/*
    echo "✅ Published to pypi.org"
    echo "   Install: pip install syrin"
fi
