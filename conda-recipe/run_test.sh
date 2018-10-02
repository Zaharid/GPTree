#!/bin/bash

zkdtree-build -i test_data/rdata.txt -n 1 -o r.tree -l 0.1 -s 0.0001

pytest

mkdir build_test
cd build_test
meson -Dbuildtype=debug -Db_sanitize=address
meson test
