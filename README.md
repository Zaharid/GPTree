# `zkdtree`


`GPTree` (`zkdtree`) is a library for quickly interpolating points in a
multidimensional space.

At the moment, it uses simple [Gaussian
Processes](http://www.gaussianprocess.org/gpml/) with an [RBF
kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel).

The library allows to perform training and inference efficiently by using
spatial data structures and suitable nearest neighbour approximations.
Specifically inference scales logarithmically with the number of training
samples, rather than cubically; the cubic scaling is instead with the much
smaller number of nearest neighbours. Contrary to the approach in
[[Shen, Ng, Seeger
2005]](https://papers.nips.cc/paper/2835-fast-gaussian-process-regression-using-kd-trees.pdf),
the algorithm adopted here results in a stable inversion of the covariance matrix
irrespective of the correlation length.


## What is in here

  - A C++ shared library with the definition of the KDTree and tools to saving
	the tree to disk.
  - A [C
    interface](https://github.com/Zaharid/GPTree/blob/master/include/capi.h) to
  - the C++ code.  A Python interface to the C interface, using Cython.

While C++ is used for development, C headers, which are much easier to integrate
with third party code are sufficient for embedding the grids for the purposes of
making predictions.

## Building

A binary [conda](https://docs.conda.io/en/latest/) package with all the
dependencies can be obtained with
[conda-build](https://docs.conda.io/projects/conda-build/en/latest/) by running

```
conda build conda-recipe
```

Manual builds can be accomplished by satisfying the relevant dependencies below
and by following the steps described in
[conda-recipe/build.sh](https://github.com/Zaharid/GPTree/blob/master/conda-recipe/build.sh):

```bash
mkdir build
cd build
meson -Dbuildtype=release -Dprefix=${PREFIX}
ninja install
```

## Dependencies

### Build system

The project is build and tested using

  - [Meson](https://mesonbuild.com)
  - [Ninja](https://ninja-build.org/)

### Build dependencies

  - A modern C++ compiler, capable of handling C++17.
  - A [LAPACK](https://en.wikipedia.org/wiki/LAPACK) provider.

### Runtime dependencies

  - A [LAPACK](https://en.wikipedia.org/wiki/LAPACK) provider.


### Dependencies for the Python interface

  - [Python](https://www.python.org/)
  - [Cython](https://cython.org/)
  - [Numpy](https://numpy.org/)

## Testing

See
[conda-recipe/run_test.sh](https://github.com/Zaharid/GPTree/blob/master/conda-recipe/run_test.sh);
essentially one can run
```
meson test
```
inside the build directory.
