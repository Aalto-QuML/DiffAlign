import sys
from pathlib import Path
import os

# Get project root
try:
    project_root = str(Path(__file__).parent.parent)
except NameError:
    project_root = str(Path.cwd().parent)

# Add to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# CRITICAL: Also set PYTHONPATH so child processes inherit it
os.environ['PYTHONPATH'] = project_root + os.pathsep + os.environ.get('PYTHONPATH', '')