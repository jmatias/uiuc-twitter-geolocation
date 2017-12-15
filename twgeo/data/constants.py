from os import path, makedirs

_working_dir = path.dirname(path.abspath(__file__))
_root_dir = path.dirname(path.dirname(_working_dir))
DATACACHE_DIR = path.join(_root_dir, ".data_cache")
DATASETS_DIR = path.join(_root_dir, "datasets")

if not path.isdir(DATACACHE_DIR):
    makedirs(DATACACHE_DIR)

if not path.isdir(DATASETS_DIR):
    makedirs(DATASETS_DIR)
