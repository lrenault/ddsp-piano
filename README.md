# DDSP-Piano: Differentiable Piano model for MIDI-to-Audio Performance Synthesis
| [**Audio Samples 🔈**](http://recherche.ircam.fr/anasyn/renault/DAFx22)
| [**DAFx Conference Paper 📄**](https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in22_paper_48.pdf)
| [**JAES Article 📄**](https://doi.org/10.17743/jaes.2022.0102) 
|

DDSP-Piano is a piano sound synthesizer from MIDI based on [DDSP](https://github.com/magenta/ddsp).

![v2.0 Architecture](assets/4_ddsp-piano_v2.drawio.svg)

## Installation
This code relies on the official Tensorflow implementation of [DDSP](https://github.com/magenta/ddsp) (tested on v3.2.0 and v3.7.0) without additional package required.
```bash
pip install --upgrade ddsp==3.7.0
```

## Audio Synthesis from MIDI

### Single MIDI file Synthesis
A piano MIDI file can be synthesized using the command:
```bash
python synthesize_midi_file.py <input_midi_file.mid> <output_file.wav>
```
Additional arguments for the inference script include:
- `-c`, `--config`: a `.gin` configuration file of a DDSP-Piano model architecture. You can chose one of the configs in the `ddsp_piano/configs/` folder.
- `--ckpt`: a checkpoint folder with your own model weights.
- `--piano_type`: the desired model among the 10 piano years learned from the MAESTRO dataset (`0` to `9`).
- `-d`, `--duration`: the maximum duration of the synthesized file. It is set by default to `None`, which will synthesize the whole file.
- `-wu`, `--warm_up`: duration of recurrent layers warm-up (to avoid undesirable noise at the beginning of the synthesized audio).
- `-u`, `--unreverbed`: toggle it to also get the dry piano sound, without reverb applying.
- `-n`, `--normalize`: set the loudness of the output file to this amount of dBFS. Set by default to `None`, which does not apply any gain modification.

The default arguments will synthesize using the most recent version of DDSP-Piano.
If you want to use the default model presented in the published papers, the inference script should look like:
```bash
python synthesize_midi_file.py \
    --config ddsp_piano/configs/dafx22.gin \
    --ckpt ddsp_piano/model_weights/dafx22/ \
    <input_midi_file.mid> <output_file.wav>
```

### Synthesize multiple performances in batch
If you want to synthesize multiple performances from MAESTRO at once, you can gather their information into a `.csv` file (see `assets/tracks_listening_test.csv` for example) and use this script:
```bash
python synthesize_from_csv.py <path/to/maestro-v3.0.0/> <your/file.csv> <output/directory/>
```
It has the same additional arguments as the `synthesize_midi_file.py` script, with the exception of `-dc` replacing the `-u` flag in order to get the dry audio, but also the isolated filtered noise and additive synthesizers outputs. 

### Evaluation script
Evaluation of the model can be conducted on the full MAESTRO test set with the corresponding script:
```bash
python evaluate_model.py <path/to/maestro-v3.0.0/> <output-directory/>
```
Additional arguments include:
- `-c`, `--config`: the `.gin` model config file.
- `--ckpt`: checkpoint to load weights from.
- `-wu`, `--warm_up`: the warm-up duration.
- `-w`, `--get_wav`: if toggled, will also save the audio of all synthesis examples.

## Model training
The paper model is trained and evaluated on the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro#v300) (v3.0.0).
After following the instructions for downloading it, a DDSP-Piano model can be trained using one of the scripts presented below.

### Dataset preprocessing (optional)
The model uses a particular encoding for handling MIDI data.
During training, conversion on the fly can take some time, on top of resampling audio data.

The following script can be used to preprocess the MIDI and audio data of MAESTRO, and store them in TFRecord format for faster data pipeline processing:
```bash
python preprocess_maestro.py <path/to/maestro-v3.0.0/> <store/tfrecords/in/this/folder/>
```
Additional arguments include:
- `-sr`: the desired audio sampling rate, to adjust accordingly to the model configuration. Set by default to 24kHz.
- `-fr`: the MIDI control frame rate, set by default to 250Hz.
- `-p`: the polyphonic capacity of the model, or maximum number of simultaneous notes handlable.

### Single Phase Training
According to our conducted listening test, decent synthesis quality can be achieved with only a single training phase, using the following python script:
```bash
python train_single_phase.py <path/to/maestro-v3.0.0/> <experiment-directory/>
```
Additional arguments include:
- `-c`, `--config`: a `.gin` model configuration file.
- `--val_path`: optional path to the `.tfrecord` file containing the preprocessed validation data (see above).
- `--batch_size`, `--steps_per_epoch`, `--epochs`, `--lr`: your usual training hyper-parameters.
- `-p`, `--phase`: the current training phase (which toggles the trainability of the corresponding layers).
- `-r`, `--restore`: a checkpoint folder to restore weights from.

Note that the `path/to/maestrov3.0.0/` can either be the extracted Maestro dataset as is, or the `maestro_training.tfrecord` preprocessed version obtained from the previous section.

During training, the Tensorboard logs are saved under `<experiment-directory>/logs/`.‡

### Full Training Procedure (legacy)
This script reproduces the full training of the default model presented in the papers:
```bash
source train_ddsp_piano.sh <path-to-maestro-v3.0.0/> <experiment-directory/>
```
It alternates between 2 training phases (one for the layers involved in the partial frequencies computation and the other for the remaining layers).
The final model checkpoint should be located in `<experiment-directory>/phase_3/last_iter/`.

However, as frequency estimation with differentiable oscillators is still an unsolved issue (see [here](https://arxiv.org/abs/2012.04572) and [here](https://doi.org/10.3389/frsip.2023.1284100)), the second training phase does not improve the model quality and we recommend to just use the single training phase script above for simplicity.

## TODO
- [x] Format code for FDN-based reverb.
- [ ] Use filtered noise synth with dynamic size on all model configs + adapt all model buildings.
- [ ] Release script for extracting single note partials estimation.
- [ ] Remove training phase related code.

## Bibtex
If you use this code for your research, please cite it as:
```latex
@article{renault2023ddsp_piano,
  title={DDSP-Piano: A Neural Sound Synthesizer Informed by Instrument Knowledge},
  author={Renault, Lenny and Mignot, Rémi and Roebel, Axel},
  journal={Journal of the Audio Engineering Society},
  volume={71},
  number={9},
  pages={552--565}
  year={2023},
  month={September}
}
```
or
```latex
@inproceedings{renault2022diffpiano,
  title={Differentiable Piano Model for MIDI-to-Audio Performance Synthesis},
  author={Renault, Lenny and Mignot, Rémi and Roebel, Axel},
  booktitle={Proceedings of the 25th International Conference on Digital Audio Effects},
  year={2022}
}
```
## Acknowledgments
This project is conducted at IRCAM and has been funded by the European Project [AI4Media](https://www.ai4media.eu/) (grant number 951911).

Thanks to @phvial for its implementation of the FDN-based reverb, in the context of the [AQUA-RIUS](https://anr.fr/Projet-ANR-22-CE23-0022) ANR project.

<p align="center">
  <a href="https://www.stms-lab.fr/"> <img src="assets/STMS-lab.png" width="9%"></a>
  &nbsp;
  <a href="https://www.ircam.fr/"> <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTeJBTYBkufxwA0Us3JCcPAlGt0blX4LzRq1QxRb27xvQ&s" width="30%"></a>
  &nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/fr/c/cd/Logo_Sorbonne_Universit%C3%A9.png" width="25%">
  &nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/fr/thumb/7/72/Logo_Centre_national_de_la_recherche_scientifique_%282023-%29.svg/512px-Logo_Centre_national_de_la_recherche_scientifique_%282023-%29.svg.png" width="10%">
  &nbsp;
  <a href="http://www.idris.fr/jean-zay/"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Logo-ministere-de-la-culture.png/788px-Logo-ministere-de-la-culture.png" width="12%"></a>
  &nbsp;
  <a href="https://ai4media.eu"> <img src="https://www.ai4europe.eu/sites/default/files/2021-06/Logo_AI4Media_0.jpg" width="14.5%"> </a>
</p>