import os

COMPLEX_TESTS = os.getenv("COMPLEX_TESTS_ENABLED", "true").lower() == "true"
