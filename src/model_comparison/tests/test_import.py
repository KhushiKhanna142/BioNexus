import unittest
import sys

class TestImport(unittest.TestCase):
    def test_sys_info(self):
        print("\n=== SYS INFO ===")
        print("EXECUTABLE:", sys.executable)
        print("VERSION:", sys.version)
