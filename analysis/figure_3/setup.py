import sys
from os import makedirs
from os.path import join, dirname, realpath, exists

current_dir = dirname(dirname(realpath(__file__)))
sys.path.insert(0, dirname(current_dir))
#from config_path import PLOTS_PATH
PLOTS_PATH = "~/_plots"
saving_dir = join(PLOTS_PATH, 'figure3')
if not exists(saving_dir):
    makedirs(saving_dir)
