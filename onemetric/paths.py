"""Where the code reads and writes. ROOT holds results/, external/ and data/; it is the repository folder unless ONEMETRIC_ROOT is set."""
import os
ROOT = os.path.abspath(os.environ.get("ONEMETRIC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
