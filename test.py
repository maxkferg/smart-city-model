"""
Test each of the modules
Modules can be selected using command line arguments
Runs the test.py file in module directory
"""
import sys
import modules.environment.test
import modules.camera.test

if sys.version_info[0] < 3:
    print('You need to run this with Python 3')
    sys.exit(1)


print("Running module tests")
#modules.environment.test.run()
modules.camera.test.run()