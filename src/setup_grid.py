from distutils.core import setup
#from Cython.Distutils import Extension
from Cython.Distutils import build_ext
from distutils.extension import Extension
from Cython.Build import cythonize
#python setup.py build_ext -i

setup(
    name="cython grid",
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize([Extension("generate_full_grid",
                                ["generate_full_grid.pyx"],
                                extra_compile_args=['-march=native'])
                            ],
                                annotate=True
                            )
    )
