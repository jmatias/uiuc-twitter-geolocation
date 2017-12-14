from os import path

_working_dir = path.dirname(path.abspath(__file__))
_root_dir = path.dirname(_working_dir)
DATACACHE_DIR = path.join(_root_dir, ".data_cache/")
DATASETS_DIR = path.join(_root_dir, "datasets/")
