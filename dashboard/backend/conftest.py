import sys
import os

# Add backend/ to sys.path so tests can import main, s3_client, etc. as siblings
sys.path.insert(0, os.path.dirname(__file__))
