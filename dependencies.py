'''
A simple module for fetching and installing dependencies
to a local subdirectory.
'''

import os
import subprocess
import sys

def assert_pkgs(pkgs):
	pkg_dir = os.getcwd() + '/packages'
	os.makedirs(pkg_dir, exist_ok=True)
	sys.path.append(pkg_dir)
	if 'PYTHONPATH' in os.environ:
		os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + os.pathsep + pkg_dir
	else:
		os.environ['PYTHONPATH'] = pkg_dir
	pkgs_installed = False
	for pkg, pkg_name in pkgs.items():
		try:
			__import__(pkg)
		except:
			if not pkgs_installed:
				print ('Installing dependencies to %s' % pkg_dir)
				pkgs_installed = True
			# add --system parameter as ubuntu sets --user as default, which conflicts with -t
			subprocess.call([sys.executable, '-m', 'pip', 'install', '--system', '-t', pkg_dir, pkg_name])

def check_pkgs():
    assert_pkgs({
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'cvi': 'cvi',
        'glob': 'glob3',
        're': 'regex'
    })

