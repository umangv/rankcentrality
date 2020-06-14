from pathlib import Path
import sys

rootdir = str((Path(__file__).parent / "..").resolve())

if rootdir not in sys.path:
    sys.path.append(rootdir)
