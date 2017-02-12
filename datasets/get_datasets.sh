#!/usr/bin/env bash
#
# This script fetches CIFAR10 dataset.

set -o nounset

readonly cifar10="cifar-10-python.tar.gz"

# Operate in the `/datasets` directory
cd $(readlink -f $(dirname $0))

# Download cifar10
init() {
    trap teardown EXIT
    wget http://www.cs.toronto.edu/~kriz/${cifar10}
}

# Remove archived cifar10
teardown() {
    rm -f ${cifar10}
}

# Extract cifar10
main() {
    tar -xzf ${cifar10}
}

init
main
