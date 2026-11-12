"""Compatibility import for the public scientific core."""

import sys

from legacy_streamlit import core

sys.modules[__name__] = core
