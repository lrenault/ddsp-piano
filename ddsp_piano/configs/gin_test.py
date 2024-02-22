import argparse
import gin
from ddsp.training.models import get_model
from ddsp_piano.data_pipeline import get_dummy_data

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('config')
	args = parser.parse_args()

	gin.parse_config_file(args.config)
	gin.bind_parameter('%inference', True)
	model = get_model()
	x = get_dummy_data(batch_size=6, duration=3)
	y = model(x)
	model.summary()
	print(model.inharm_model.beta_t.get_weights())

	from ddsp.training import trainers, train_util
	strategy = train_util.get_strategy()
	with strategy.scope():
		model = get_model()
		trainer = trainers.Trainer(model=model, strategy=strategy)
	trainer.build(get_dummy_data(batch_size=1,
						   	     duration=1,
								 sample_rate=model.sample_rate))
	print(model.inharm_model.beta_t.get_weights())
	import pdb; pdb.set_trace()