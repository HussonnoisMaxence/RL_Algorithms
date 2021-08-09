import yaml
import numpy as np


def read_seed():
	with open("./seed.txt",'r') as filin:
		for line in filin:
			seed=int(line)

		filin.close()
	return seed

def write_seed(seed):
	with open("./seed.txt",'w') as fileout:
		fileout.write(str(seed))

		fileout.close()
		   


def read_config(config_path):
	with open(config_path, "r") as ymlfile:
		cfg = yaml.load(ymlfile, yaml.FullLoader)

		ymlfile.close()
	return cfg
