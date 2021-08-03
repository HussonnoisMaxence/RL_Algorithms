import yaml, sys

def read_config(config_path):
	with open(config_path, "r") as ymlfile:
		cfg = yaml.load(ymlfile, yaml.FullLoader)

	return cfg