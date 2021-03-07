


def get_trainer(name, args, datas, log_dir, logger, tensorboard_logger):
    if name == 'SPADE':
        from .ConditionImageSynthesis.SemanticLabel2Image.SPADE.models.trainer import SPADETrainer
        return SPADETrainer(args, datas, log_dir, logger, tensorboard_logger)
    elif name == 'bottlelayout':
        from .ConditionImageSynthesis.SceneGraph2Image.BottleLayout.models.trainer import Sg2ImBottleLayoutTrainer
        return Sg2ImBottleLayoutTrainer(args, datas, log_dir, logger, tensorboard_logger)
    elif name == 'cyclelayout':
        from .ConditionImageSynthesis.SceneGraph2Image.CycleLayout.models.trainer import Sg2ImCycleLayoutTrainer
        return Sg2ImCycleLayoutTrainer(args, datas, log_dir, logger, tensorboard_logger)
    elif name == 'sketch_transformer':
        from .SketchTransformer.models.trainer import SketchTransformerTrainer
        return SketchTransformerTrainer(args, datas, log_dir, logger, tensorboard_logger)
    elif name == 'sketch_transformer_test':
        from .SketchTransformer.models.trainer_test import SketchTransformerTrainer
        return SketchTransformerTrainer(args, datas, log_dir, logger, tensorboard_logger)
    elif name == 'sketch_transformer_sbir':
        from .SketchTransformer.models.sbir_trainer import SketchTransformerSBIRTrainer
        return SketchTransformerSBIRTrainer(args, datas, log_dir, logger, tensorboard_logger)
    elif name == 'sketch_segment_transformer':
        from .SketchTransformer.models.segment_trainer import SketchSegmentTransformerTrainer
        return SketchSegmentTransformerTrainer(args, datas, log_dir, logger, tensorboard_logger)
    elif name == 'sketchrnn':
        from .SketchRNN.models.trainer import SketchRNNTrainer
        return SketchRNNTrainer(args, datas, log_dir, logger, tensorboard_logger)
    elif name == 'sketchcnn':
        from .SketchCNN.models.trainer import SketchCNNTrainer
        return SketchCNNTrainer(args, datas, log_dir, logger, tensorboard_logger)
    elif name == 'sketchrnncls':
        from .SketchRNN.models.cls_trainer import SketchClsRNNTrainer
        return SketchClsRNNTrainer(args, datas, log_dir, logger, tensorboard_logger)
    elif name == 'sketchrnnsbir':
        from .SketchRNN.models.sbir_trainer import SketchRNNSBIRTrainer
        return SketchRNNSBIRTrainer(args, datas, log_dir, logger, tensorboard_logger)
    elif name == 'sketchcnnsbir':
        from .SketchCNN.models.sbir_trainer import SketchCNNSBIRTrainer
        return SketchCNNSBIRTrainer(args, datas, log_dir, logger, tensorboard_logger)
    elif name == 'sketch_transformer_vae':
        from .SketchTransformer.models.vae_trainer import SketchTransformerVAETrainer
        return SketchTransformerVAETrainer(args, datas, log_dir, logger, tensorboard_logger)
    elif name == 'sketch_transformer_gan':
        from .SketchTransformer.models.gan_trainer import SketchTransformerGANTrainer
        return SketchTransformerGANTrainer(args, datas, log_dir, logger, tensorboard_logger)
