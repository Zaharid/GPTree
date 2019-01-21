project('kdtree', 'cpp', default_options: ['cpp_std=c++17'])

sources = ['tree.hpp', ]

lib = shared_library('kdtree',
sources: sources,
install_dir: 'lib/',
install: true,
)

test_sources = ['run_test.cpp',] + sources

test_exe = executable('run_test', test_sources,
    build_by_default:false)

test('run_test', test_exe)
