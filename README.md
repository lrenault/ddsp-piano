# ddsp-piano
DDSP-Piano is a piano sound synthesizer from MIDI based on [DDSP](https://github.com/magenta/ddsp)

### Links
- Paper: "Differentiable Piano Model for MIDI-to-Audio Performance Synthesis", DAFx20in22 (link TBA) 📄
- [Some audio examples](http://recherche.ircam.fr/anasyn/renault/DAFx22) 🔈

## Installation
This code relies on the official tensorflow implementation of [DDSP](https://github.com/magenta/ddsp) (tested on v3.2.0) without additional package required.
```
pip install --upgrade ddsp==3.2.0
```

## Inference
```
python synthesize_midi_file.py <a_midi_file.mid>
```

## Model training
The paper model is trained with the [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset (v3.0.0).

After following the instructions for downloading the dataset, a default piano model can be trained with the shell script:
```
train_ddsp_piano.sh <path-to-maestro-v3.0.0/> <export-directory>
```
Default parameters reproduces the full training of the paper model, but training for a single phase can be done with `training_single_phase.py`.


## Bibtex
```
@inproceedings{
	renault2022diffpiano,
	title={Differentiable Piano Model for MIDI-to-Audio Performance Synthesis},
	author={Lenny Renault and Rémi Mignot and Axel Roebel},
	booktitle={International Conference on Digital Audio Effects}
	year={2022}
}
```