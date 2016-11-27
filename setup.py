

from distutils.core import setup, Extension

setup(name='MultiNEAT',
      version='0.1',
      py_modules=['MultiNEAT'],
      ext_modules=[Extension('_MultiNEAT', [
                                            'lib/EvolvableSubstrate.cpp',
	                                    'lib/Genome.cpp',
                                            'lib/Innovation.cpp',
                                            'lib/Main.cpp',
                                            'lib/NeuralNetwork.cpp',
                                            'lib/Parameters.cpp',
                                            'lib/Point.cpp',
                                            'lib/PhenotypeBehavior.cpp',
                                            'lib/Population.cpp',
                                            'lib/PythonBindings.cpp',
					    'lib/SubstrateBase.cpp',
                                            'lib/Random.cpp',
                                            'lib/Species.cpp',
                                            'lib/Substrate.cpp',
                                            'lib/Utils.cpp'],
					    include_dirs=['lib'],
					    extra_compile_args=['-std=c++11'],
                             libraries=['boost_python',
                                        'boost_serialization'])]
      )
