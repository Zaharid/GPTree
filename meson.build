project('kdtree', 'cpp', default_options: ['cpp_std=c++17'])

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

test('run_test',
    test_exe,
	timeout: 60,
)
