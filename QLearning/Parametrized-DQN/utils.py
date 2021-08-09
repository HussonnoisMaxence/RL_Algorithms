import yaml
import random
import numpy as np
import sys


def read_config(config_path):
	with open(config_path, "r") as ymlfile:
		cfg = yaml.load(ymlfile, yaml.FullLoader)

	return cfg

def change_config(config_path, key, value):
	print(key)
	with open(config_path) as f:
	     list_doc = yaml.load(f)
	list_doc[key] = value

	with open(config_path, "w") as f:
	    yaml.dump(list_doc, f)

