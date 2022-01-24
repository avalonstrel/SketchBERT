# Implementation of Sketch-BERT: Learning Sketch Bidirectional Encoder Representation from Transformers by Self-supervised Learning of Sketch Gestalt[https://arxiv.org/abs/2005.09159]
Hangyu Lin, Yanwei Fu, Xiangyang Xue, Yu-Gang Jiang

## Abstract
Previous researches of sketches often considered sketches in pixel format and leveraged CNN based models in the sketch understanding. Fundamentally, a sketch is stored as a sequence of data points, a vector format representation, rather than the photo-realistic image of pixels. SketchRNN studied a generative neural representation for sketches of vector format by Long Short Term Memory networks (LSTM). Unfortunately, the representation learned by SketchRNN is primarily for the generation tasks, rather than the other tasks of recognition and retrieval of sketches. To this end and inspired by the recent BERT model , we present a model of learning Sketch Bidirectional Encoder Representation from Transformer (Sketch-BERT). We generalize BERT to sketch domain, with the novel proposed components and pre-training algorithms, including the newly designed sketch embedding networks, and the self-supervised learning of sketch gestalt. Particularly, towards the pre-training task, we present a novel Sketch Gestalt Model (SGM) to help train the Sketch-BERT. Experimentally, we show that the learned representation of Sketch-BERT can help and improve the performance of the downstream tasks of sketch recognition, sketch retrieval, and sketch gestalt.

## Pre-train Model
You can get the pretrained model on [Google Drive](https://drive.google.com/file/d/1y6-0RqzdqrExDkHC0BXOzIRUEl_Ei1da/view?usp=sharing).

## Preparing the Dataset
To efficient loading the datast, I will first change the original dataset into [memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) format.

Here I will take quickdraw dataset as an example to show how to use the script to generate quickdraw of memmap.

To use the script `models/SketchTransformer/models/generate_dataset.py`
You need first generate a txt file with all [npz](https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn) file path in your server like this,

------
path/beach.full.npz

path/cat.full.npz

...

...

path/dog.full.npz

------

And then setting the path information in the pyhton script to generate your dataset of memmap. 

## Test with pre-trained model

Just run `bash scripts/test_sketch_transformer.sh` .
The corresponding config file is models/SketchTransformer/config/sketch_transformer.yml

You need to set several parameters to successfully run the code.

------
**task_types: ['maskrec'] # ['maskrec' 'sketchclsinput', 'sketchretrieval']**

You can choose one of them ['maskrec' 'sketchclsinput', 'sketchretrieval'], 'maskrec' means pre-training, 'sketchclsinput' means classification, 
'sketchretrieval' means retrieval.

------

**load_pretrained: 'scratch' # ['scratch', 'continue', 'pretrained']**

**which_pretrained: ['enc_net'] # ['enc_net', 'enc_opt', 'task_net', 'task_opt']**

**restore_checkpoint_path: 'qd_8_12_768.pth.tar'**

The loading settings, if you want train from scratch just setting 'scratch', if you want continue training from some checkpoint just use 'continue', if you just want to load a pre-trained weight but with other things like optimizer initialized, use 'pretrained'.

which_pretrained is used to setting load which part of the network, 'enc_net' means loading the sketchbert encoder(transformer part) and 'task_net' means loading the cls or retrieval head. 'enc_opt' and 'task_opt' means to load the optimizers weights.

restore_checkpoint_path should be the path to pre-trained weight in your server.

------

**sum_path: 'QuickDraw/memmap_sum.txt'** 

**offset_path: 'QuickDraw/offsets.npz'**

This part refers to the dataset information. Just see the instruction in Preparing the Dataset.

------

**log_dir: 'sketch_albert_qd_struct_8_12_768_test_mask_max_scale'**

The save tag for your experiment. Some checkpoints or results will be save in this dir.

## Training your own model

Just run `bash scripts/sketch_transformer.sh`
The corresponding config file is models/SketchTransformer/config/sketch_transformer.yml

Except for the parameters above, you may want to specify some new structure. Some important parameters are shown below.

------
**max_length: 250**

The max length of the model, should adjust to your data.

**layers_setting: [[12, 768, 3072],...,[12, 768, 3072]] (repeat 8 times)**

The layer setting for the transformer, you can set the L-A-H inside and define the layer by repeating it.

**embed_layers_setting: [128,256,512]**

The layer setting for embedding network, just the hidden sizes of fully-connected networks.

**rec_layers_setting: [512,256,128]**

The layer setting for reconstruction network, just the hidden sizes of fully-connected networks.

------
## Acknowledge
Thanks to the [QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) Dataset and the [BERT](https://github.com/google-research/bert).
