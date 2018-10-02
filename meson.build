project('kdtree', ['c', 'cpp'], default_options: ['cpp_std=c++17'])

sources = ['tree.hpp', ]

vendor_includes = include_directories('vendor')

dependencies = [
    dependency('lapack'),
]

lib = shared_library('kdtree',
	sources: sources,
	dependencies: dependencies,
	include_directories: vendor_includes,
	install_dir: 'lib/',
	install: true,
)

cinc = include_directories('include')
install_headers(['include/capi.h'], subdir: 'zkdtree')


clib = shared_library('capi',
	sources: ['capi.cpp'],
	dependencies: dependencies,
	include_directories: [cinc, vendor_includes],
	install_dir: 'lib/',
	install: true,
)


test_sources = ['run_test.cpp',] + sources

test_exe = executable('run_test', test_sources,
    include_directories: vendor_includes,
    dependencies: dependencies,
    build_by_default:false,
)

build_exe = executable('zkdtree-build', 'zkdtree_build.cpp',
    include_directories: vendor_includes,
	dependencies: dependencies,
	install: true,
)

desc_exe = executable('zkdtree-describe', 'zkdtree_describe.cpp',
    include_directories: vendor_includes,
	dependencies: dependencies,
	install: true,
)

test('run_test',
    test_exe,
	timeout: 60,
)

pkg = import('pkgconfig')
pkg.generate(
	libraries: clib,
	subdirs: ['.'],
	install_dir: 'lib',
	name: 'libzkdtree',
	filebase: 'zkdtree',
	description: 'A grid interpolation library',
)

cython = find_program('cython')

pyx_c = custom_target('pyzkdtree.',
  output : 'pyzkdtree.c',
  input : 'pyzkdtree.pyx',
  command : [cython, '-3' , '@INPUT@', '-o', '@OUTPUT@'],
)

py3_mod = import('python')

python = dependency('python3')

python_inst = py3_mod.find_installation('python3')

numpy_includes = run_command(
	python_inst, '-c', 'import numpy; print(numpy.get_include())',
	check:true
).stdout().strip()

message(numpy_includes)


pymod = python_inst.extension_module('pyzkdtree',
  pyx_c,
  dependencies: [python,],
  link_with : clib,
  install: true,
  include_directories: [cinc, numpy_includes],
  install_dir: python_inst.get_install_dir()
)
