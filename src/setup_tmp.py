from distutils.core import setup
#from Cython.Distutils import Extension
from Cython.Distutils import build_ext
from distutils.extension import Extension
import cython_gsl
from Cython.Build import cythonize
#python setup.py build_ext -i

setup(
    #[...]
    name="cython prediction using gsl",
    #ext_modules=cythonize(extensions),
    include_dirs = [cython_gsl.get_include()],
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize([Extension("prediction_over_grid",
                                ["prediction_over_grid.pyx"],
                                libraries=cython_gsl.get_libraries(),
                                library_dirs=[cython_gsl.get_library_dir()],
                                include_dirs=[cython_gsl.get_cython_include_dir()],
                                extra_compile_args=['-fopenmp'],
                                extra_link_args=['-fopenmp'])
                            ],
                                annotate=True
                            )
    )
