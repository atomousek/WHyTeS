from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import cython_gsl
from Cython.Build import cythonize
#python setup_integration.py build_ext -i

setup(
    #[...]
    name="cython integration using gsl",
    #ext_modules=cythonize(extensions),
    include_dirs = [cython_gsl.get_include()],
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize([Extension("integration",
                                ["integration.pyx"],
                                libraries=cython_gsl.get_libraries(),
                                library_dirs=[cython_gsl.get_library_dir()],
                                include_dirs=[cython_gsl.get_cython_include_dir()])],
                                annotate=True)
    )
