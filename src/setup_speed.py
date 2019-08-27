from distutils.core import setup
#from Cython.Distutils import Extension
from Cython.Distutils import build_ext
from distutils.extension import Extension
import cython_gsl
from Cython.Build import cythonize
#python setup_speed.py build_ext -i

setup(
    name="cython speed",
    include_dirs = [cython_gsl.get_include()],
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize([Extension("speed_over_angles",
                                ["speed_over_angles.pyx"],
                                extra_compile_args=['-march=native'])
                            ],
                                annotate=True
                            )
    )
