#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json, os

from .utils import imagenet_preprocess, imagenet_deprocess
from .utils import imagenet_deprocess_batch
from .quickdraw_dataset import QuickDrawMemMapDataset, QuickDrawSBIRDataset, QuickDrawImageDataset, QuickDrawImageSBIRDataset
from .tuberlin_dataset import TUBerlinMemMapDataset, TUBerlinSBIRDataset, TUBerlinImageDataset, TUBerlinImageSBIRDataset
from torch.utils.data import Dataset, DataLoader

def build_vg_dsets(args, logger):
    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)
    dset_kwargs = {
        'vocab': vocab,
        'h5_path': args.train_h5,
        'image_dir': args.vg_image_dir,
        'image_size': args.image_size,
        'max_samples': args.num_train_samples,
        'max_objects': args.max_objects_per_image,
        'use_orphaned_objects': args.vg_use_orphaned_objects,
        'include_relationships': args.include_relationships,
        'sketch_type':args.sketch_type,
    }
    train_dset = VisualGenomeDataset(**dset_kwargs)
    iter_per_epoch = len(train_dset) // args.batch_size
    logger.info('There are %d iterations per epoch' % iter_per_epoch)
    args.num_classes = train_dset.num_classes
    dset_kwargs['h5_path'] = args.val_h5
    del dset_kwargs['max_samples']
    val_dset = VisualGenomeDataset(**dset_kwargs)

    return vocab, train_dset, val_dset, iter_per_epoch

def build_coco_dsets(args, logger):
    dset_kwargs = {
        'image_dir': args.coco_train_image_dir,
        'instances_json': args.coco_train_instances_json,
        'stuff_json': args.coco_train_stuff_json,
        'stuff_only': args.coco_stuff_only,
        'image_size': args.image_size,
        'mask_size': args.mask_size,
        'max_samples': args.num_train_samples,
        'min_object_size': args.min_object_size,
        'min_objects_per_image': args.min_objects_per_image,
        'max_objects_per_image': args.max_objects_per_image,
        'instance_whitelist': args.instance_whitelist,
        'stuff_whitelist': args.stuff_whitelist,
        'include_other': args.coco_include_other,
        'include_relationships': args.include_relationships,
    }
    train_dset = CocoSceneGraphDataset(**dset_kwargs)
    num_objs = train_dset.total_objects()
    num_imgs = len(train_dset)
    iter_per_epoch = len(train_dset) // args.batch_size
    logger.info('There are %d iterations per epoch' % iter_per_epoch)
    logger.info('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
    logger.info('(%.2f objects per image)' % (float(num_objs) / num_imgs))

    dset_kwargs['image_dir'] = args.coco_val_image_dir
    dset_kwargs['instances_json'] = args.coco_val_instances_json
    dset_kwargs['stuff_json'] = args.coco_val_stuff_json
    dset_kwargs['max_samples'] = args.num_val_samples
    val_dset = CocoSceneGraphDataset(**dset_kwargs)
    num_objs = val_dset.total_objects()
    num_imgs = len(val_dset)
    logger.info('Val dataset has %d images and %d objects' % (num_imgs, num_objs))
    logger.info('(%.2f objects per image)' % (float(num_objs) / num_imgs))

    assert train_dset.vocab == val_dset.vocab
    vocab = json.loads(json.dumps(train_dset.vocab))

    return vocab, train_dset, val_dset,iter_per_epoch

def build_cmplaces_dsets(args, logger):
    dset_kwargs = {
        'data_dir':args.cmplaces_train_image_dir,
        'img_flist_path':args.cmplaces_train_flist_path,
        'image_size':args.image_size,
    }
    train_dset = CMPlacesDataset(**dset_kwargs)
    args.num_classes = train_dset.num_classes
    iter_per_epoch = len(train_dset) // args.batch_size
    logger.info('There are %d iterations per epoch' % iter_per_epoch)

    dset_kwargs['data_dir'] = args.cmplaces_val_image_dir
    dset_kwargs['img_flist_path'] = args.cmplaces_val_flist_path

    val_dset = CMPlacesDataset(**dset_kwargs)

    return None, train_dset, val_dset, iter_per_epoch

def build_coco_cis_dsets(args, logger):
    dset_kwargs = {
        'img_dir': args.coco_train_image_dir,
        'instances_json': args.coco_train_instances_json,
        'img_size': args.img_size,
        'max_samples': args.num_train_samples,
    }
    train_dset = CocoCISDataset(**dset_kwargs)
    num_imgs = len(train_dset)
    iter_per_epoch = len(train_dset) // args.batch_size
    logger.info('There are %d iterations per epoch' % iter_per_epoch)
    logger.info('Training dataset has %d images.' % (num_imgs))

    dset_kwargs['img_dir'] = args.coco_val_image_dir
    dset_kwargs['instances_json'] = args.coco_val_instances_json
    dset_kwargs['max_samples'] = args.num_val_samples
    val_dset = CocoCISDataset(**dset_kwargs)
    num_imgs = len(val_dset)
    logger.info('Val dataset has %d images.' % (num_imgs))

    return None, train_dset, val_dset, iter_per_epoch

def build_quickdraw_image_dsets(args, logger):
    dset_kwargs = {
         'image_dir':args.sum_path,
         'mode':args.mode,
         'get_type':args.get_type,
         'image_size':args.image_size,
         'max_cls_cache':args.max_cls_cache,
         'each_max_samples':args.each_max_samples
    }
    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_train_sum.txt')
    train_dset = QuickDrawImageDataset(**dset_kwargs)
    num_imgs = len(train_dset)
    iter_per_epoch = len(train_dset) // args.batch_size
    logger.info('There are %d iterations per epoch' % iter_per_epoch)
    logger.info('Training dataset has %d images.' % (num_imgs))

    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_valid_sum.txt')
    dset_kwargs['mode'] = 'valid'
    dset_kwargs['each_max_samples'] = args.each_val_samples
    val_dset = QuickDrawImageDataset(**dset_kwargs)
    num_imgs = len(val_dset)
    logger.info('Val dataset has %d images.' % (num_imgs))

    return None, train_dset, val_dset, iter_per_epoch

def build_quickdraw_image_sbir_dsets(args, logger):
    dset_kwargs = {
         'sketch_sum_path':args.sketch_sum_path,
         'image_sum_path':args.image_sum_path,
         'mode':args.mode,
         'get_type':args.get_type,
         'image_size':args.image_size,
         'max_cls_cache':args.max_cls_cache,
         'each_max_samples':args.each_max_samples,
         'each_image_max_samples':args.each_image_max_samples,
    }
    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_train_sum.txt')
    train_dset = QuickDrawImageSBIRDataset(**dset_kwargs)
    num_imgs = len(train_dset)
    iter_per_epoch = len(train_dset) // args.batch_size
    logger.info('There are %d iterations per epoch' % iter_per_epoch)
    logger.info('Training dataset has %d images.' % (num_imgs))

    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_valid_sum.txt')
    dset_kwargs['mode'] = 'valid'
    dset_kwargs['each_max_samples'] = args.each_val_samples
    val_dset = QuickDrawImageSBIRDataset(**dset_kwargs)
    num_imgs = len(val_dset)
    logger.info('Val dataset has %d images.' % (num_imgs))

    return None, train_dset, val_dset, iter_per_epoch

def build_tuberlin_image_dsets(args, logger):
    dset_kwargs = {
         'image_sum_path':args.sum_path,
         'mode':args.mode,
         'get_type':args.get_type,
         'image_size':args.image_size,
         'max_cls_cache':args.max_cls_cache,
         'each_max_samples':args.each_max_samples
    }
    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_train_sum.txt')
    train_dset = TUBerlinImageDataset(**dset_kwargs)
    num_imgs = len(train_dset)
    iter_per_epoch = len(train_dset) // args.batch_size
    logger.info('There are %d iterations per epoch' % iter_per_epoch)
    logger.info('Training dataset has %d images.' % (num_imgs))

    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_valid_sum.txt')
    dset_kwargs['mode'] = 'valid'
    dset_kwargs['each_max_samples'] = args.each_val_samples
    val_dset = TUBerlinImageDataset(**dset_kwargs)
    num_imgs = len(val_dset)
    logger.info('Val dataset has %d images.' % (num_imgs))

    return None, train_dset, val_dset, iter_per_epoch

def build_tuberlin_image_sbir_dsets(args, logger):
    dset_kwargs = {
         'sketch_sum_path':args.sketch_sum_path,
         'image_sum_path':args.image_sum_path,
         'mode':args.mode,
         'get_type':args.get_type,
         'image_size':args.image_size,
         'max_cls_cache':args.max_cls_cache,
         'each_max_samples':args.each_max_samples,
         'each_image_max_samples':args.each_image_max_samples,
    }
    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_train_sum.txt')
    train_dset = TUBerlinImageSBIRDataset(**dset_kwargs)
    num_imgs = len(train_dset)
    iter_per_epoch = len(train_dset) // args.batch_size
    logger.info('There are %d iterations per epoch' % iter_per_epoch)
    logger.info('Training dataset has %d images.' % (num_imgs))

    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_valid_sum.txt')
    dset_kwargs['mode'] = 'valid'
    dset_kwargs['each_max_samples'] = args.each_val_samples
    val_dset = TUBerlinImageSBIRDataset(**dset_kwargs)
    num_imgs = len(val_dset)
    logger.info('Val dataset has %d images.' % (num_imgs))

    return None, train_dset, val_dset, iter_per_epoch

def build_quickdraw_memmap_dsets(args, logger):
    dset_kwargs = {
        'sum_path':args.sum_path,
        'offset_path':args.offset_path,
        'mode': args.mode,
        'task_types':args.task_types,
        'get_type':args.get_type,
        'max_length':args.max_length,
        'max_size':args.max_size,
        'type_size':args.type_size,
        'mask_prob':args.mask_prob,
        'limit':args.limit,
        'stroke_type':args.stroke_type,
        'input_is_complete':args.input_is_complete,
        'mask_task_type':args.mask_task_type,
        'normalization_type': args.normalization_type,
        'max_scale_factor':args.max_scale_factor,
        'max_cls_cache':args.max_cls_cache,
        'each_max_samples':args.each_max_samples,
        'cls_limit_path':args.cls_limit_path
    }
    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_train_sum.txt')
    train_dset = QuickDrawMemMapDataset(**dset_kwargs)
    num_imgs = len(train_dset)
    iter_per_epoch = len(train_dset) // args.batch_size
    logger.info('There are %d iterations per epoch' % iter_per_epoch)
    logger.info('Training dataset has %d images.' % (num_imgs))

    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_valid_sum.txt')
    dset_kwargs['mode'] = 'valid'
    dset_kwargs['each_max_samples'] = args.each_val_samples
    val_dset = QuickDrawMemMapDataset(**dset_kwargs)
    num_imgs = len(val_dset)
    logger.info('Val dataset has %d images.' % (num_imgs))

    return None, train_dset, val_dset, iter_per_epoch

def build_quickdraw_sbir_dsets(args, logger):
    dset_kwargs = {
        'sketch_sum_path':args.sum_path,
        'sketch_offset_path':args.offset_path,
        'image_sum_path':args.image_sum_path,
        'mode': args.mode,
        'task_types':args.task_types,
        'get_type':args.get_type,
        'max_length':args.max_length,
        'max_size':args.max_size,
        'image_size':args.image_size,
        'type_size':args.type_size,
        'mask_prob':args.mask_prob,
        'limit':args.limit,
        'stroke_type':args.stroke_type,
        'input_is_complete':args.input_is_complete,
        'mask_task_type':args.mask_task_type,
        'normalization_type': args.normalization_type,
        'max_scale_factor':args.max_scale_factor,
        'max_cls_cache':args.max_cls_cache,
        'each_max_samples':args.each_max_samples,
        'each_image_max_samples':args.each_image_max_samples,
        'cls_limit_path':args.cls_limit_path
    }
    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_train_sum.txt')
    train_dset = QuickDrawSBIRDataset(**dset_kwargs)
    num_imgs = len(train_dset)
    iter_per_epoch = len(train_dset) // args.batch_size
    logger.info('There are %d iterations per epoch' % iter_per_epoch)
    logger.info('Training dataset has %d images.' % (num_imgs))

    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_valid_sum.txt')
    dset_kwargs['mode'] = 'valid'
    dset_kwargs['each_max_samples'] = args.each_val_samples
    val_dset = QuickDrawSBIRDataset(**dset_kwargs)

    num_imgs = len(val_dset)
    logger.info('Val dataset has %d images.' % (num_imgs))

    return None, train_dset, val_dset, iter_per_epoch


def build_sketchy_image_dsets(args, logger):
    dset_kwargs = {
         'image_sum_path':args.sum_path,
         'mode':args.mode,
         'get_type':args.get_type,
         'image_size':args.image_size,
         'max_cls_cache':args.max_cls_cache,
         'each_max_samples':args.each_max_samples
    }
    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_train_sum.txt')
    train_dset = SketchyImageDataset(**dset_kwargs)
    num_imgs = len(train_dset)
    iter_per_epoch = len(train_dset) // args.batch_size
    logger.info('There are %d iterations per epoch' % iter_per_epoch)
    logger.info('Training dataset has %d images.' % (num_imgs))

    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_valid_sum.txt')
    dset_kwargs['mode'] = 'valid'
    dset_kwargs['each_max_samples'] = args.each_val_samples
    val_dset = SketchyImageDataset(**dset_kwargs)
    num_imgs = len(val_dset)
    logger.info('Val dataset has %d images.' % (num_imgs))

    return None, train_dset, val_dset, iter_per_epoch

def build_sketchy_image_sbir_dsets(args, logger):
    dset_kwargs = {
         'sketch_sum_path':args.sketch_sum_path,
         'image_sum_path':args.image_sum_path,
         'mode':args.mode,
         'get_type':args.get_type,
         'image_size':args.image_size,
         'max_cls_cache':args.max_cls_cache,
         'each_max_samples':args.each_max_samples,
         'each_image_max_samples':args.each_image_max_samples,
    }
    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_train_sum.txt')
    train_dset = SketchyImageSBIRDataset(**dset_kwargs)
    num_imgs = len(train_dset)
    iter_per_epoch = len(train_dset) // args.batch_size
    logger.info('There are %d iterations per epoch' % iter_per_epoch)
    logger.info('Training dataset has %d images.' % (num_imgs))

    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_valid_sum.txt')
    dset_kwargs['mode'] = 'valid'
    dset_kwargs['each_max_samples'] = args.each_val_samples
    val_dset = SketchyImageSBIRDataset(**dset_kwargs)
    num_imgs = len(val_dset)
    logger.info('Val dataset has %d images.' % (num_imgs))

    return None, train_dset, val_dset, iter_per_epoch

def build_sketchy_memmap_dsets(args, logger):
    dset_kwargs = {
        'sum_path':args.sum_path,
        'offset_path':args.offset_path,
        'mode': args.mode,
        'task_types':args.task_types,
        'get_type':args.get_type,
        'max_length':args.max_length,
        'max_size':args.max_size,
        'type_size':args.type_size,
        'mask_prob':args.mask_prob,
        'limit':args.limit,
        'stroke_type':args.stroke_type,
        'input_is_complete':args.input_is_complete,
        'mask_task_type':args.mask_task_type,
        'normalization_type': args.normalization_type,
        'max_scale_factor':args.max_scale_factor,
        'max_cls_cache':args.max_cls_cache,
        'each_max_samples':args.each_max_samples,
        'cls_limit_path':args.cls_limit_path
    }
    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_train_sum.txt')
    train_dset = SketchyMemMapDataset(**dset_kwargs)
    num_imgs = len(train_dset)
    iter_per_epoch = len(train_dset) // args.batch_size
    logger.info('There are %d iterations per epoch' % iter_per_epoch)
    logger.info('Training dataset has %d images.' % (num_imgs))

    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_valid_sum.txt')
    dset_kwargs['mode'] = 'valid'
    dset_kwargs['each_max_samples'] = args.each_val_samples
    val_dset = SketchyMemMapDataset(**dset_kwargs)
    num_imgs = len(val_dset)
    logger.info('Val dataset has %d images.' % (num_imgs))

    return None, train_dset, val_dset, iter_per_epoch

def build_sketchy_sbir_dsets(args, logger):
    dset_kwargs = {
        'sketch_sum_path':args.sum_path,
        'sketch_offset_path':args.offset_path,
        'image_sum_path':args.image_sum_path,
        'mode': args.mode,
        'task_types':args.task_types,
        'get_type':args.get_type,
        'max_length':args.max_length,
        'max_size':args.max_size,
        'image_size':args.image_size,
        'type_size':args.type_size,
        'mask_prob':args.mask_prob,
        'limit':args.limit,
        'stroke_type':args.stroke_type,
        'input_is_complete':args.input_is_complete,
        'mask_task_type':args.mask_task_type,
        'normalization_type': args.normalization_type,
        'max_scale_factor':args.max_scale_factor,
        'max_cls_cache':args.max_cls_cache,
        'each_max_samples':args.each_max_samples,
        'each_image_max_samples':args.each_image_max_samples,
        'cls_limit_path':args.cls_limit_path
    }
    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_train_sum.txt')
    train_dset = SketchySBIRDataset(**dset_kwargs)
    num_imgs = len(train_dset)
    iter_per_epoch = len(train_dset) // args.batch_size
    logger.info('There are %d iterations per epoch' % iter_per_epoch)
    logger.info('Training dataset has %d images.' % (num_imgs))

    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_valid_sum.txt')
    dset_kwargs['mode'] = 'valid'
    dset_kwargs['each_max_samples'] = args.each_val_samples
    val_dset = SketchySBIRDataset(**dset_kwargs)

    num_imgs = len(val_dset)
    logger.info('Val dataset has %d images.' % (num_imgs))

    return None, train_dset, val_dset, iter_per_epoch



def build_tuberlin_memmap_dsets(args, logger):
    dset_kwargs = {
        'sum_path':args.sum_path,
        'offset_path':args.offset_path,
        'mode': args.mode,
        'task_types':args.task_types,
        'get_type':args.get_type,
        'max_length':args.max_length,
        'max_size':args.max_size,
        'type_size':args.type_size,
        'mask_prob':args.mask_prob,
        'limit':args.limit,
        'stroke_type':args.stroke_type,
        'input_is_complete':args.input_is_complete,
        'mask_task_type':args.mask_task_type,
        'normalization_type': args.normalization_type,
        'max_scale_factor':args.max_scale_factor,
        'max_cls_cache':args.max_cls_cache,
        'each_max_samples':args.each_max_samples,
        'cls_limit_path':args.cls_limit_path
    }
    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_train_sum.txt')
    train_dset = TUBerlinMemMapDataset(**dset_kwargs)
    num_imgs = len(train_dset)
    iter_per_epoch = len(train_dset) // args.batch_size
    logger.info('There are %d iterations per epoch' % iter_per_epoch)
    logger.info('Training dataset has %d images.' % (num_imgs))

    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_valid_sum.txt')
    dset_kwargs['mode'] = 'valid'
    dset_kwargs['each_max_samples'] = args.each_val_samples
    val_dset = TUBerlinMemMapDataset(**dset_kwargs)
    num_imgs = len(val_dset)
    logger.info('Val dataset has %d images.' % (num_imgs))

    return None, train_dset, val_dset, iter_per_epoch

def build_tuberlin_sbir_dsets(args, logger):
    dset_kwargs = {
        'sketch_sum_path':args.sum_path,
        'sketch_offset_path':args.offset_path,
        'image_sum_path':args.image_sum_path,
        'mode': args.mode,
        'task_types':args.task_types,
        'get_type':args.get_type,
        'max_length':args.max_length,
        'max_size':args.max_size,
        'image_size':args.image_size,
        'type_size':args.type_size,
        'mask_prob':args.mask_prob,
        'limit':args.limit,
        'stroke_type':args.stroke_type,
        'input_is_complete':args.input_is_complete,
        'mask_task_type':args.mask_task_type,
        'normalization_type': args.normalization_type,
        'max_scale_factor':args.max_scale_factor,
        'max_cls_cache':args.max_cls_cache,
        'each_max_samples':args.each_max_samples,
        'each_image_max_samples':args.each_image_max_samples,
        'cls_limit_path':args.cls_limit_path
    }
    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_train_sum.txt')
    train_dset = TUBerlinSBIRDataset(**dset_kwargs)
    num_imgs = len(train_dset)
    iter_per_epoch = len(train_dset) // args.batch_size
    logger.info('There are %d iterations per epoch' % iter_per_epoch)
    logger.info('Training dataset has %d images.' % (num_imgs))

    #dset_kwargs['sum_path'] = os.path.join(args.sum_path, 'memmap_valid_sum.txt')
    dset_kwargs['mode'] = 'valid'
    dset_kwargs['each_max_samples'] = args.each_val_samples
    val_dset = TUBerlinSBIRDataset(**dset_kwargs)

    num_imgs = len(val_dset)
    logger.info('Val dataset has %d images.' % (num_imgs))

    return None, train_dset, val_dset, iter_per_epoch


def build_loaders(args, logger):
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': True,
        'pin_memory':False
    }

    if args.dataset == 'vg':
        vocab, train_dset, val_dset, iter_per_epoch = build_vg_dsets(args, logger)
        loader_kwargs['collate_fn'] = vg_collate_fn
    elif args.dataset == 'coco':
        vocab, train_dset, val_dset, iter_per_epoch = build_coco_dsets(args, logger)
        loader_kwargs['collate_fn'] = coco_collate_fn
    elif args.dataset == 'cmplaces':
        vocab, train_dset, val_dset, iter_per_epoch = build_cmplaces_dsets(args, logger)
    elif args.dataset == 'coco_cis':
        vocab, train_dset, val_dset, iter_per_epoch = build_coco_cis_dsets(args, logger)
    elif args.dataset == 'quickdraw_memmap':
        vocab, train_dset, val_dset, iter_per_epoch = build_quickdraw_memmap_dsets(args, logger)
    elif args.dataset == 'quickdraw_image':
        vocab, train_dset, val_dset, iter_per_epoch = build_quickdraw_image_dsets(args, logger)
    elif args.dataset == 'quickdraw_image_sbir':
        vocab, train_dset, val_dset, iter_per_epoch = build_quickdraw_image_sbir_dsets(args, logger)
    elif args.dataset == 'quickdraw_sbir':
        vocab, train_dset, val_dset, iter_per_epoch = build_quickdraw_sbir_dsets(args, logger)
    elif args.dataset == 'sketchy_image':
        vocab, train_dset, val_dset, iter_per_epoch = build_sketchy_image_dsets(args, logger)
    elif args.dataset == 'sketchy_image_sbir':
        vocab, train_dset, val_dset, iter_per_epoch = build_sketchy_image_sbir_dsets(args, logger)
    elif args.dataset == 'sketchy_memmap':
        vocab, train_dset, val_dset, iter_per_epoch = build_sketchy_memmap_dsets(args, logger)
    elif args.dataset == 'sketchy_sbir':
        vocab, train_dset, val_dset, iter_per_epoch = build_sketchy_sbir_dsets(args, logger)
    elif args.dataset == 'tuberlin_image':
        vocab, train_dset, val_dset, iter_per_epoch = build_tuberlin_image_dsets(args, logger)
    elif args.dataset == 'tuberlin_image_sbir':
        vocab, train_dset, val_dset, iter_per_epoch = build_tuberlin_image_sbir_dsets(args, logger)
    elif args.dataset == 'tuberlin_memmap':
        vocab, train_dset, val_dset, iter_per_epoch = build_tuberlin_memmap_dsets(args, logger)
    elif args.dataset == 'tuberlin_sbir':
        vocab, train_dset, val_dset, iter_per_epoch = build_tuberlin_sbir_dsets(args, logger)

    train_loader = DataLoader(train_dset, **loader_kwargs)
    loader_kwargs['shuffle'] = args.shuffle_val
    val_loader = DataLoader(val_dset, **loader_kwargs)
    train_check_loader,val_check_loader = [],[]

    for i, batch in enumerate(train_loader):
        train_check_loader.append(batch)
        if i >= args.num_val_samples / args.batch_size:
            break
    for i, batch in enumerate(val_loader):
        val_check_loader.append(batch)
        if i >= args.num_val_samples / args.batch_size:
            break

    return vocab, train_loader, train_check_loader, val_loader, val_check_loader, iter_per_epoch
