from setuptools import setup

setup(name='simulatION',
      version='0.4.0',
      description='Simulates Oxford Nanopore Technologies MinION reads',
      url='https://git.rohreich.de/projects/DNA/repos/nanopore_simulation/',
      author='FH Kiel',
      author_email='christian.rohrandt@fh-kiel.de',
      license='MIT',
      packages=['simulation'],
      install_requires=["BioPython", "pandas", "h5py", "scipy", "matplotlib", "argparse", "numpy"],
      scripts=["bin/simulatION"],
      include_package_data=True,
      zip_safe=False)
