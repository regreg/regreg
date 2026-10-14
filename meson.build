project('regreg', 'c', 'cython',
  version : run_command('tools/get_version.py', check : true).stdout().strip(),
  license : 'BSD-2-Clause',
  meson_version : '>=1.1.0',
  default_options : ['warning_level=1', 'buildtype=release'])

py = import('python').find_installation(pure : false)
py_dep = py.dependency()

# NumPy headers for the Cython extensions
incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check : true).stdout().strip()
np_dep = declare_dependency(include_directories : include_directories(incdir_numpy))

cython_args = ['-3']

# Cython extension modules
foreach ext : [['projl1_cython', 'regreg/atoms'],
               ['mixed_lasso_cython', 'regreg/atoms'],
               ['piecewise_linear', 'regreg/atoms']]
  py.extension_module(
    ext[0],
    ext[1] / ext[0] + '.pyx',
    cython_args : cython_args,
    dependencies : [py_dep, np_dep],
    install : true,
    subdir : ext[1]
  )
endforeach

py.extension_module(
  'cox_utils',
  ['regreg/smooth/cox_utils.pyx', 'regreg/smooth/cox_fns.c'],
  cython_args : cython_args,
  include_directories : include_directories('regreg/smooth'),
  dependencies : [py_dep, np_dep],
  install : true,
  subdir : 'regreg/smooth'
)

# Version file, generated from the git tag at build time
configure_file(
  input : 'regreg/_version.py.in',
  output : '_version.py',
  configuration : {'VERSION' : meson.project_version()},
  install : true,
  install_dir : py.get_install_dir() / 'regreg'
)

# Install Python sources
install_subdir(
  'regreg',
  install_dir : py.get_install_dir(),
  exclude_directories : ['__pycache__', '.ipynb_checkpoints'],
  exclude_files : ['.gitignore', '_version.py.in']
)
