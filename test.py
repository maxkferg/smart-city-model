"""
Test each of the modules
Modules can be selected using command line arguments
Runs the test.py file in module directory
"""
import sys
import numpy as np
import tensorflow as tf

print("Using numpys version ==", np.__version__)
print("Using tensorflow version ==", tf.__version__)
print("Running module tests")

import modules.environment.test
import modules.camera.test
import modules.ssd.test

if sys.version_info[0] < 3:
    print('You need to run this with Python 3')
    sys.exit(1)

import tensorflow as tf
tf.__version__

#modules.environment.test.run()
modules.camera.test.run()
modules.ssd.test.run()