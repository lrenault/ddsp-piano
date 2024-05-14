import argparse
import gin
from ddsp.training import trainers, train_util
from ddsp.training.models import get_model
from ddsp_piano.data_pipeline import get_dummy_data


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('config', help="A .gin config file to test.")
	args = parser.parse_args()

	gin.parse_config_file(args.config)
	gin.bind_parameter('%inference', True)
	model = get_model()
	x = get_dummy_data(batch_size=6, duration=3)
	y = model(x)
	model.summary()

	strategy = train_util.get_strategy()
	with strategy.scope():
		model = get_model()
		model.alternate_training(first_phase=True)
		trainer = trainers.Trainer(model=model, strategy=strategy)
	trainer.build(get_dummy_data(batch_size=1,
						   	     duration=3,
								 sample_rate=model.sample_rate))

	import pdb; pdb.set_trace()