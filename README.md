# DDSP-Piano: Differentiable Piano model for MIDI-to-Audio Performance Synthesis
DDSP-Piano is a piano sound synthesizer from MIDI based on [DDSP](https://github.com/magenta/ddsp).

### Links
- Paper, published in DAFx20in22 (link TBA) 📄
- [Some audio examples](http://recherche.ircam.fr/anasyn/renault/DAFx22) 🔈

## Installation
This code relies on the official tensorflow implementation of [DDSP](https://github.com/magenta/ddsp) (tested on v3.2.0) without additional package required.
```
pip install --upgrade ddsp==3.2.0
```

## Inference
A piano MIDI file can be synthesized by the model with the command:
```
python synthesize_midi_file.py <input_midi_file.mid> <output_file.wav>
```
Additional arguments for the inference script include:
- `--piano_model`: the desired model among the 10 piano years learned from the MAESTRO dataset.
- `--ckpt`: a path to a different checkpoint from the provided weights.

## Model training
The paper model is trained with the [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset (v3.0.0).

After following the instructions for downloading the dataset, a default piano model can be trained with the shell script:
```
train_ddsp_piano.sh <path-to-maestro-v3.0.0/> <experiment-directory>
```
This training configuration reproduces the full training of the model presented in the paper, but training for a single phase can be done with the python script `train_single_phase.py`.


## Bibtex
If you use this code, please cite it as:
```
@inproceedings{
  renault2022diffpiano,
  title={Differentiable Piano Model for MIDI-to-Audio Performance Synthesis},
  author={Lenny Renault and Rémi Mignot and Axel Roebel},
  booktitle={International Conference on Digital Audio Effects}
  year={2022}
}
```