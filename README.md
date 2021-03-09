Implementation of Sketch-BERT: Learning Sketch Bidirectional Encoder Representation from Transformers by Self-supervised Learning of Sketch Gestalt[https://arxiv.org/abs/2005.09159]

Abstract:Previous researches of sketches often considered sketches in pixel format and leveraged CNN based models in the sketch understanding. Fundamentally, a sketch is
stored as a sequence of data points, a vector format representation, rather than the photo-realistic image of pixels. SketchRNN [7] studied a generative neural representation for sketches of vector format by Long Short Term Memory networks (LSTM). Unfortunately, the representation learned by SketchRNN is primarily for the generation
tasks, rather than the other tasks of recognition and retrieval of sketches. To this end and inspired by the recent
BERT model [3], we present a model of learning Sketch Bidirectional Encoder Representation from Transformer
(Sketch-BERT). We generalize BERT to sketch domain,
with the novel proposed components and pre-training
algorithms, including the newly designed sketch embedding
networks, and the self-supervised learning of sketch gestalt.
Particularly, towards the pre-training task, we present a
novel Sketch Gestalt Model (SGM) to help train the Sketch-
BERT. Experimentally, we show that the learned representation
of Sketch-BERT can help and improve the performance
of the downstream tasks of sketch recognition, sketch
retrieval, and sketch gestalt.
