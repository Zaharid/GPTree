#!/bin/bash
set -u
set -v


mkdir build
cd build
meson -Dbuildtype=release -Dprefix=${PREFIX}
meson test
ninja install
