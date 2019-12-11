[![DOI](https://zenodo.org/badge/218984364.svg)](https://zenodo.org/badge/latestdoi/218984364)

# `zkdtree`


`GPTree` (`zkdtree`) is a library for quickly interpolating points in a
multidimensional space.

At the moment, it uses simple [Gaussian
Processes](http://www.gaussianprocess.org/gpml/) with an [RBF
kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel).

The library allows to perform training and inference efficiently by using
spatial data structures and suitable nearest neighbour approximations.
Specifically inference scales logarithmically with the number of training
samples, rather than cubically as in the naive implementation; the cubic scaling
is instead with the much smaller number of nearest neighbours. Contrary to the
approach in
[[Shen, Ng, Seeger
2005]](https://papers.nips.cc/paper/2835-fast-gaussian-process-regression-using-kd-trees.pdf),
the algorithm adopted here results in a stable inversion of the covariance matrix
irrespective of the correlation length.

## Summary of the grid construction algorithm

The input data consists on a `N x M` matrix corresponding the `M` coordinates of
the `N` data points at which the target function is evaluated and `y`, the
vector of values of the function at each data point.

The data is firstly structured in a [k-d
tree](https://en.wikipedia.org/wiki/K-d_tree). The implementation is inspired by
that in
[scikit-learn](https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/neighbors),
but has some added simplifications and optimizations for this use case. For
example the data is reordered physically in order to avoid an extra level of
indirection and improve cache locality.

A file (*grid*) containing the tree and hyperparameters is exported using the
[cereal](https://uscilab.github.io/cereal/) library.

Inference for a point `x_test` is then achieved by first finding the
`nneighbours` (which is itself an hyperparmeter) closest point closest points to
`x_test`, (`X_nearest`, `y_nearest`)

```
y_test = K(x_test, X_nearest)@inverse(K(X_nearest, X_rearest) + noise@Identity)@y_nearest
```
where `K` is the two point Kernel function and `noise` is the noise
hyperparmeter. Currently the inversion of the `nneighbours x nneighbours` linear
system is done on the fly for each point, since this is efficient enough for the
problems this code has been used for, when using the
[appropriate](https://software.intel.com/en-us/mkl-developer-reference-fortran-ppsv)
LAPACK routine.

An estimate of the uncertainty at the test point can also be
provided by the library.


Hyperparameter estimation can be achieved using a [Leave One
Out](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-one-out_cross-validation)
cross validation based method, for example with the help of the [Python
interface](https://github.com/Zaharid/GPTree/blob/ff7f4ddb7ddd7df07a80fa06dc00358ee47724ef/pyzkdtree.pyx#L58).
More facilities will be provided in the future.

## What is in here

  - A C++ shared library with the definition of the KDTree and tools to saving
	the tree to disk.
  - A [C
    interface](https://github.com/Zaharid/GPTree/blob/master/include/capi.h) to
    the C++ code.
  - A Python interface to the C interface, using Cython.
  - Executables to build a grid file from a text file and to query basic
	properties of the grid.

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

## Integrating grid predictions

The C interface is the recommended way of integrating grid predictions into
third party software. In C itself, it would look like

```c
#include <stdio.h>
#include <stdlib.h>
#include <zkdtree/capi.h>

int main(){
	char *err;
	zkdtree* t = zkdtree_load("r.tree", &err);
	if(!t){
		printf("Error: %s", err);
		free(err);
		return 1;
	}
	double x[1] = {0};
	printf("Result at x=0 is %f", zkdtree_interpolate(t, x));
	zkdtree_delete(t);
	return 0;

}
```

A [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/) file
specifying the required build options is generated as a part of the
installation. This allows required compiler flags to be easily queried with
build systems such as CMake or Meson. The following Meson configuration suffices
to build the file above

```meson
project('zkdtreetest', 'c')
deps = [dependency('zkdtree')]
exe = executable('main', 'main.c', dependencies:deps)
```
