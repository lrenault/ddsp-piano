import argparse
import gin
from ddsp.training.models import get_model

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('config')
	args = parser.parse_args()

	gin.parse_config_file(args.config)
	gin.bind_parameter('%inference', True)
	model = get_model()
	import pdb; pdb.set_trace()