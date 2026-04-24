export CC=cc
export FC=gfortran
export F77=gfortran
export F90=gfortran

export BLAS_LIBS=" "
export LAPACK_LIBS=" "
export SCALAPACK_LIBS=" "
export FFT_LIBS=" "

export LDFLAGS="-L/opt/homebrew/opt/openblas/lib"
export CPPFLAGS="-I/opt/homebrew/opt/openblas/include"
export CFLAGS="-fPIC -march=native"
export CXXFLAGS="-fPIC -march=native"
export CMAKE_CXX_FLAGS="-fPIC -march=native"

export NPY_USE_BLAS_ILP64=1
export NPY_DISABLE_SVML=1

openblas_install_path=/opt/homebrew/opt/openblas

export CMAKE_PREFIX_PATH=${openblas_install_path}/lib:$CMAKE_PREFIX_PATH
export CMAKE_PREFIX_PATH=${openblas_install_path}/include:$CMAKE_PREFIX_PATH
export PHONO3PY_USE_CMAKE=false

source ./.venv_dl/bin/activate
