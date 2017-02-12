#!/usr/bin/env bash
#
# This script fetches CIFAR10 dataset.

set -o nounset

readonly cifar10="cifar-10-python.tar.gz"

# Operate in the `/datasets` directory
cd $(readlink -f $(dirname $0))

# Download cifar10
init() {
    # trap teardown EXIT
    if [[ ! -e "./${cifar10}" ]]; then
        wget http://www.cs.toronto.edu/~kriz/${cifar10}
    fi
}

# Remove archived cifar10
teardown() {
    rm -rf ${cifar10}
}

# Extract cifar10
main() {
    tar -xzf ${cifar10} --keep-old-files
}

init
main
