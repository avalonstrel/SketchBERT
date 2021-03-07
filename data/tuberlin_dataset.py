import json, os, random, math
import PIL
import torch

import numpy as np
import pickle as pkl
import torchvision.transforms as T


from torch.utils.data import Dataset, DataLoader
from models.SketchTransformer.models.utils import *
from .utils import resize_strokes
STROKE_DIMS = {'stroke-6':6, 'stroke-5':5, 'stroke-4':4, 'stroke-3':3, 'stroke-discrete':3}


def get_class(path):
    return path[path.rfind('/')+1:-4]
class ImageDataset(Dataset):
    def __init__(self, image_list, cls_list, image_size):
        self.image_list = image_list
        self.cls_list = cls_list
        self.image_size = image_size
        self._make_transform()

    def _make_transform(self):

        size = self.image_size
        self._image_transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
        ])
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(self.image_list[index])
        return self._image_transform(image), self.cls_list[index]

class TUBerlinImageDataset(Dataset):

    """
    Dataset Class for QuickDraw Dataset
    Params:

    Return:
        Corresponding Scene Graph and Images
    """
    def __init__(self, image_sum_path, mode, image_size, get_type, max_cls_cache, each_max_samples):
        self.image_sum_path = image_sum_path
        self.mode = mode
        self.get_type = get_type
        self.image_size = image_size
        self.max_cls_cache = max_cls_cache
        self.each_max_samples = each_max_samples
        self._make_transform()
        self.load_dataset()

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        if 'triplet' in self.get_type:
            return self.get_triplet(index)
        else:
            return self.get_term(index)

    def get_term(self, index):
        image_path = self.image_paths[index]
        cls_ind = self.classes[index]
        image = Image.open(image_path)
        image = np.array(image)
        image = image.reshape(image.shape+(1,))

        image = np.tile(image, (1,1,3))
        image = Image.fromarray(image.astype(np.uint8))
        image = self._image_transform(image)
        #print(image.size())
        return image, cls_ind

    def get_triplet(self, index):
        query_image, cls = self.get_term(index)
        pos_cls, neg_clses = cls, [i for i in self.cls2index if i != cls]
        #print(pos_cls, neg_clses)
        neg_cls = np.random.choice(neg_clses, 1)[0]

        pos_index = np.random.choice([i for i in self.cls2index[pos_cls] if i != index],1)[0]
        neg_index = np.random.choice(self.cls2index[neg_cls],1)[0]
        pos_image, pos_cls = self.get_term(pos_index)
        neg_image, neg_cls = self.get_term(neg_index)
        return query_image, pos_image, neg_image, cls, pos_cls, neg_cls
    def get_cate_num(self):
        return {c:len(self.cls2index[c]) for c in self.cls2index}

    def _make_transform(self):

        size = self.image_size
        self._image_transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
        ])

    def load_dataset(self):

        image_sum_path = self.image_sum_path
        self.image_paths = []
        self.classes = []
        self.cls2name = {}
        self.name2cls = {}
        self.cls2index = {}
        cls_i, ind = -1, 0

        with open(image_sum_path, 'r') as rf:
            for line in rf:
                path, cls_name, mode = line.strip().split('\t')
                if cls_i >= self.max_cls_cache:
                    break
                if cls_name not in self.name2cls:
                    cls_i += 1
                    self.cls2name[cls_i] = cls_name
                    self.cls2index[cls_i] = []
                    self.name2cls[cls_name] = cls_i


                if len(self.cls2index[cls_i]) >= self.each_max_samples or mode != self.mode:
                    continue
                self.image_paths.append(path)
                self.classes.append(self.name2cls[cls_name])
                self.cls2index[self.name2cls[cls_name]].append(ind)
                ind += 1
        self.num = len(self.image_paths)


class TUBerlinImageSBIRDataset(Dataset):
    """
    Dataset Class for QuickDraw Dataset
    Params:

    Return:
        Corresponding Scene Graph and Images
    """
    def __init__(self, sketch_sum_path, image_sum_path, mode, image_size, get_type, max_cls_cache, each_max_samples, each_image_max_samples):
        self.sketch_sum_path = sketch_sum_path
        self.image_sum_path = image_sum_path
        self.mode = mode
        self.get_type = get_type
        self.image_size = image_size
        self.max_cls_cache = max_cls_cache
        self.each_max_samples = each_max_samples
        self.each_image_max_samples = each_image_max_samples
        self._make_transform()
        self.load_sketch_dataset()
        self.load_image_dataset()

        print('cls_num',len(self.sketch_cls2index.keys()), len(self.image_cls2index.keys()))
    def __len__(self):
        return self.sketch_num

    def __getitem__(self, index):
        return self.get_triplet(index)


    def get_term(self, index):
        image_path = self.sketch_paths[index]
        cls_ind = self.sketch_classes[index]
        image = Image.open(image_path)
        image = np.array(image)
        image = image.reshape(image.shape+(1,))

        image = np.tile(image, (1,1,3))
        image = Image.fromarray(image.astype(np.uint8))
        image = self._image_transform(image)
        #print(image.size())
        return image, cls_ind

    def get_image_term(self, index):
        image_path = self.image_paths[index]
        cls_ind = self.image_classes[index]
        image = Image.open(image_path)
        #print(np.array(image).shape)
        image = self._image_transform(image)
        #print(image.size())
        return image, cls_ind

    def get_triplet(self, index):
        query_image, cls = self.get_term(index)
        pos_cls, neg_clses = cls, [i for i in self.sketch_cls2index if i != cls]
        #print(pos_cls, neg_clses)
        neg_cls = np.random.choice(neg_clses, 1)[0]

        pos_index = np.random.choice(self.image_cls2index[pos_cls],1)[0]
        neg_index = np.random.choice(self.image_cls2index[neg_cls],1)[0]
        pos_image, pos_cls = self.get_image_term(pos_index)
        neg_image, neg_cls = self.get_image_term(neg_index)
        return query_image, pos_image, neg_image, cls, pos_cls, neg_cls
    def get_cate_num(self):
        return {c:len(self.image_cls2index[c]) for c in self.image_cls2index}

    def _make_transform(self):

        size = self.image_size
        self._image_transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
        ])

    def load_image_dataset(self):
        self.image_paths = []
        self.image_classes = []
        self.image_cls2index = {}
        ind = 0
        with open(self.image_sum_path, 'r') as rf:
            for line in rf:
                image_path, cls_name, mode = line.strip().split('\t')
                if cls_name in self.name2cls:
                    cls_index = self.name2cls[cls_name]
                    if cls_index in self.image_cls2index and len(self.image_cls2index[cls_index]) >= (int(self.mode=='train') * self.each_image_max_samples+int(self.mode=='valid')*self.each_max_samples):
                        continue
                    self.image_paths.append(image_path)
                    self.image_classes.append(cls_index)
                    if cls_index in self.image_cls2index:
                        self.image_cls2index[cls_index].append(ind)
                    else:
                        self.image_cls2index[cls_index] = [ind]
                    ind += 1
        self.image_classes = np.array(self.image_classes)
        self.image_num = len(self.image_classes)
        print('Sketch Num:{}, Image Num:{}'.format(self.sketch_num, self.image_num))

    def load_sketch_dataset(self):
        # Load Sketch Paths
        #sketch_dir = os.path.join(self.sketch_dir, self.mode)
        self.sketch_paths = []
        self.sketch_classes = []
        self.cls2name = {}
        self.sketch_cls2index = {}
        self.name2cls = {}
        cls_i, ind = -1, 0

        with open(self.sketch_sum_path, 'r') as rf:
            for line in rf:
                path, cls_name, mode = line.strip().split('\t')
                if cls_i >= self.max_cls_cache:
                    break
                if cls_name not in self.name2cls:
                    cls_i += 1
                    self.cls2name[cls_i] = cls_name
                    self.sketch_cls2index[cls_i] = []
                    self.name2cls[cls_name] = cls_i


                if len(self.sketch_cls2index[cls_i]) >= self.each_max_samples or mode != self.mode:
                    continue
                self.sketch_paths.append(path)
                self.sketch_classes.append(self.name2cls[cls_name])
                self.sketch_cls2index[self.name2cls[cls_name]].append(ind)
                ind += 1

        self.sketch_classes = np.array(self.sketch_classes)
        self.sketch_num = len(self.sketch_paths)

    def get_image_loader(self, **args):
        return DataLoader(ImageDataset(self.image_paths, self.image_classes, self.image_size), **args)

class TUBerlinMemMapDataset(Dataset):
    """
    Dataset Class for TUBerlin MemMap Dataset
    Params:
        sum_path[str]: Path to summary file, it will be used to construct the dataset
        mode[str]: 'train', 'valid', 'test'
        max_length[int]: max length for the sequence of strokes
        mask_prob[float]: the probability for mask which is used for masked Sketch Model Learning
        stroke_type[str]: The input type of strokes, now is from SketchRNN, stroke-5;stroke-3
        mask_task_type[str]: How to mask the data, can be divided in three type ['full','task','single'],
                             'full' means mask every dimenstion randomly, 'task' means mask each task each term randomly, 'single' means mask the whole term
        normalization[str]: Normalization type ['max_scale', 'var_scale']
        cls_in_input[bool]: Whether cls label in the Input which is used in classification task
        max_cls_cache[int]: Maximum class number in cache

    Return:
        A simple quickdraw simple dataset with random mask
    """

    def __init__(self, sum_path, offset_path, mode, task_types, get_type, max_length=250, max_size=[256,256], type_size=3, mask_prob=0.8, limit=1000,
                  stroke_type='stroke-5', input_is_complete=True, mask_task_type='task', normalization_type='var_scale', max_scale_factor=10, max_cls_cache=500, each_max_samples=10000, cls_limit_path=None):
        self.mode = mode
        self.task_types = task_types
        self.get_type = get_type
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.limit = limit
        self.stroke_type = stroke_type
        self.input_is_complete = input_is_complete
        self.max_cls_cache = max_cls_cache
        self.each_max_samples = each_max_samples
        self.mask_task_type = mask_task_type
        self.normalization_type = normalization_type
        self.max_scale_factor = max_scale_factor
        self.cls_in_input = 'sketchclsinput' in self.task_types
        self.rel_in_input = 'sketchretrieval' in self.task_types
        self.only_maskrec = 'maskrec' in self.task_types and 'sketchclsinput' not in self.task_types and 'sketchretrieval' not in self.task_types

        if mode == 'train' and self.only_maskrec:
            self.modes = ['train','test','valid']
        else:
            self.modes = [mode]
        self.stroke_dim = STROKE_DIMS[stroke_type]
        #print(stroke_type, self.stroke_dim)
        self.max_size = max_size
        self.type_size = type_size
        assert self.mask_task_type in ['full', 'task', 'single']
        assert self.normalization_type in ['global_scale','var_scale', 'max_scale']
        #self.each_val_samples = each_val_samples
        self.load_dataset(sum_path, offset_path, cls_limit_path)



    def __len__(self):
        return self.num

    '''
    Output:
        mask_input[batch, seq_len+cls_input, input_dim](Masked Input)
        stroke[batch, seq_len, input_dim](Original Input)
        segment[batch, seq_len+cls_in_input]
        length_mask[batch, seq_len+cls_input]
        mask[batch, seq_len+cls_input, input_dim](Random Mask)
        cls[batch] ( Class Target)
    '''

    def __getitem__(self, index):
        if self.get_type == 'triplet':
            return self.get_triplet(index)
        else:
            return self.get_term(index)

    def get_term(self, index):
        skt_ind = self.index_dict[index]
        path = self.sketch_paths[skt_ind[0]]
        sketches = np.memmap(path, dtype=np.float32, mode='r', shape=(self.offsets[self.mode][self.cls2name[skt_ind[0]]][-1][-1], 3))
        sketch = sketches[skt_ind[1][0]:skt_ind[1][1],:].copy()
        del sketches
        cls = self.classes[index]
        sketch, segment, length_mask = self.transform_stroke(sketch, self.stroke_type, self.normalization_type,  self.max_length, self.max_size, self.type_size)
        # print(sketch)
        return self.get_mask_result(sketch, segment, length_mask, cls)

    def get_mask_result(self, stroke, segment, length_mask, cls):
        length = int(torch.sum(length_mask))
        mask = self.generate_mask(stroke, length, self.max_length, self.mask_task_type, self.stroke_dim, self.mask_prob)
        mask = mask.to(dtype=stroke.dtype)
        #print(mask.size(), stroke.size(), self.stroke_dim)
        mask_input = stroke * mask
        if self.cls_in_input:
            cls_input = torch.ones(1, stroke.size(1)).to(dtype=torch.float)
            mask_input = torch.cat([(cls_input-2).type(dtype=torch.float), mask_input.type(dtype=torch.float)], dim=0)
            segment = torch.cat([(cls_input[0,0:1]-1).type(dtype=torch.long), segment.type(dtype=torch.long)], dim=0)
            length_mask = torch.cat([cls_input[0,0:1].type(dtype=torch.float), length_mask.type(dtype=torch.float)], dim=0)
        if self.rel_in_input:
            cls_input = torch.ones(1, stroke.size(1)).to(dtype=torch.float)
            mask_input = torch.cat([(cls_input-3).type(dtype=torch.float), mask_input.type(dtype=torch.float)], dim=0)
            segment = torch.cat([(cls_input[0,0:1]-1).type(dtype=torch.long), segment.type(dtype=torch.long)], dim=0)
            length_mask = torch.cat([cls_input[0,0:1].type(dtype=torch.float), length_mask.type(dtype=torch.float)], dim=0)
        # print(mask_input.size(), stroke.size(), segment.size(), length_mask.size(), mask.size())
        return mask_input, stroke, segment, length_mask, mask, cls

    def get_triplet(self, index):
        mask_input, stroke, segment, length_mask, mask, cls = self.get_term(index)
        pos_cls, neg_clses = cls, [i for i in self.cls2index if i != cls]
        #print(pos_cls, neg_clses)
        neg_cls = np.random.choice(neg_clses, 1)[0]

        pos_index = np.random.choice([i for i in self.cls2index[pos_cls] if i != index], 1)[0]
        neg_index = np.random.choice(self.cls2index[neg_cls],1)[0]
        pos_mask_input, pos_stroke, pos_segment, pos_length_mask, pos_mask, pos_cls = self.get_term(pos_index)
        neg_mask_input, neg_stroke, neg_segment, neg_length_mask, neg_mask, neg_cls = self.get_term(neg_index)
        return (mask_input, pos_mask_input, neg_mask_input), (stroke, pos_stroke, neg_stroke), (segment, pos_segment, neg_segment), (length_mask, pos_length_mask, neg_length_mask), (mask, pos_mask, neg_mask), (cls, pos_cls, neg_cls)
    def get_cate_num(self):
        return {c:len(self.cls2index[c]) for c in self.cls2index}

    def transform_stroke(self, stroke, stroke_type, normalization_type, max_length, max_size, type_size):
        # Insert prefix & Truncating the postions out of max length
        # Generate the length mask, sum it = the length
        length_mask = torch.zeros(max_length)
        length_mask[:len(stroke)] = 1
        '''
        Cls: 0
        Non meaning: 1
        Other Segments: 2-max
        '''
        # Generate the segment
        segment = torch.zeros(len(stroke))
        pre_i, now_i, stroke_i = 0, 0, 2
        for i in range(len(stroke)):
            if stroke[i, 2] == 1:
                now_i = i + 1
                segment[pre_i:now_i] = stroke_i
                stroke_i = stroke_i + 1
                pre_i = now_i
        max_segment = torch.zeros(max_length)
        truncated_length = min(max_length, len(stroke))
        max_segment[:truncated_length] = segment[:truncated_length]
        if len(stroke) < max_length:
            max_segment[len(stroke):max_length] = 1

        segment = max_segment
        # Truncate the stroke
        stroke = stroke[:max_length]
        # Change the stroke dim

        # remove the large gap
        stroke = np.minimum(stroke, self.limit)
        stroke = np.maximum(stroke, -self.limit)

        if normalization_type == 'max_scale' and not stroke_type == 'stroke-discrete':
            bounds = np.array(get_bounds(stroke, factor=1))
            sizes = bounds[[1,3]] - bounds[[0,2]]
            if sizes[0] == 0:
                sizes[0] = 1
            if sizes[1] == 0:
                sizes[1] = 1
            stroke[:,:2] = stroke[:,:2] / sizes.reshape(1,2) * self.max_scale_factor

        elif normalization_type == 'var_scale' and not stroke_type == 'stroke-discrete':
            vars = np.std(stroke[:,:2], axis=0)
            if vars[0] == 0:
                vars[0] = 1
            if vars[1] == 0:
                vars[1] = 1
            means = np.mean(stroke[:,:2], axis=0)
            stroke[:,:2] = (stroke[:,:2] - means.reshape(1,2)) / vars.reshape(1,2)

        if stroke_type == 'stroke-5':
            stroke = to_big_strokes(stroke, max_length)
        elif stroke_type == 'stroke-3':
            stroke = extend_strokes(stroke, max_length)
        elif stroke_type == 'stroke-discrete':
            stroke = to_discrete_strokes(stroke, max_length, max_size)


        return torch.from_numpy(stroke), segment, length_mask


    def generate_mask(self, stroke, length, max_length, type='full', dim=6, mask_prob=0.8):
        max_mask = torch.ones(max_length, 6)

        if type == 'single':
            mask_length = int(length * (1-mask_prob))
            mask = np.random.choice(length, length, replace=False)[:mask_length]
            max_mask[mask, :] = 0
        elif type == 'task':
            # Mask the axis
            for ind in [2,3]:
                tmp_length = int(torch.sum(stroke[:,ind]).item())

                tmp_mask_length = int(tmp_length * (1-mask_prob))
                if tmp_mask_length == 0:
                    continue
                mask = np.random.choice(tmp_length, tmp_length, replace=False)[:tmp_mask_length]
                mask_index = torch.arange(max_length)
                mask_index = mask_index[stroke[:,ind] == 1][mask]
                max_mask[mask_index, :2] = 0

            # Mask the states
            for ind in [2,3,4]:

                #print(stroke, stroke[:,dim])
                tmp_length = int(torch.sum(stroke[:,ind]).item())

                tmp_mask_length = int(tmp_length * (1-mask_prob))
                if tmp_mask_length == 0:
                    continue
                mask = np.random.choice(tmp_length, tmp_length, replace=False)[:tmp_mask_length]
                mask_index = torch.arange(max_length)
                mask_index = mask_index[stroke[:,ind] == 1][mask]
                max_mask[mask_index, 2:5] = 0
            # Always mask the true end
            max_mask[length-1,2:5] = 0
            # mask = np.random.choice(length, length, replace=False)[:mask_length]
            # max_mask[mask, 5:] = 0

        return max_mask[:, :dim]

    def load_cls_limit_path(self, cls_limit_path):
        cls_limit_list = []
        if cls_limit_path is not None and cls_limit_path != '':
            with open(cls_limit_path, 'r') as rf:
                for line in rf:
                    cls_limit_list.append(line.strip())
        return cls_limit_list
    def load_dataset(self, sum_path, offset_path, cls_limit_path):
        self.sketch_paths = []
        self.classes = []
        self.cls2name = {}
        self.index_dict = {}
        self.offsets = pkl.load(open(offset_path, 'rb'))
        self.cls2index = {}
        self.cls_limit_list = self.load_cls_limit_path(cls_limit_path)
        i, ind = 0, 0
        for tmp_mode in self.modes:
            with open(sum_path, 'r') as rf:
                for line in rf:
                    if i >= self.max_cls_cache:
                        break

                    # Unpackage
                    items = line.strip().split('\t')
                    path, num = items[0], int(items[1])
                    # Update the dict

                    cls_name = get_class(path)
                    if len(self.cls_limit_list) > 0 and cls_name not in self.cls_limit_list:
                        continue
                    print(path)
                    self.cls2name[i] = cls_name
                    # Load the memmap datas
                    # data = np.memmap(path, dtype=np.int16, mode='r', shape=(self.offsets[self.mode][self.cls2name[i]][-1][-1], 3))
                    # if self.mode == 'train':
                    #     self.mode = 'half_full'
                    # if self.mode == 'valid':
                    #     self.mode = 'test'
                    self.sketch_paths.append(path.replace('.dat', '_{}.dat'.format(tmp_mode)))
                    for j in range(min(len(self.offsets[tmp_mode][self.cls2name[i]]), self.each_max_samples)):
                        tmp_offset = self.offsets[tmp_mode][self.cls2name[i]][j]
                        if tmp_offset[1] - tmp_offset[0] >= self.max_length and self.input_is_complete:
                            continue
                        self.index_dict[ind] = (i, tmp_offset)
                        self.classes.append(i)
                        if i in self.cls2index:
                            self.cls2index[i].append(ind)
                        else:
                            self.cls2index[i] = [ind]
                        ind += 1
                    i += 1
                #print(self.mode,self.cls2name)

        self.classes = np.array(self.classes)
        # self.strokes, self.segments, self.length_masks, self.classes, self.cls2index = self.preprocess(self.sketches, self.classes)
        self.num = len(self.classes)


class TUBerlinSBIRDataset(Dataset):
    """
    Dataset Class for QuickDraw SBIR tasks
    Params:
        sum_path[str]: Path
        mode[str]: 'train', 'valid', 'test'
        max_length[int]: max length for the sequence of strokes
        mask_prob[float]: the probability for mask which is used for masked Sketch Model Learning
        stroke_type[str]: The input type of strokes, now is from SketchRNN, stroke-5;stroke-3
        mask_task_type[str]: How to mask the data, can be divided in three type ['full','task','single'],
                             'full' means mask every dimenstion randomly, 'task' means mask each task each term randomly, 'single' means mask the whole term
        normalization[str]: Normalization type ['max_scale', 'var_scale']
        cls_in_input[bool]: Whether cls label in the Input which is used in classification task
        max_cls_cache[int]: Maximum class number in cache

    Return:
        A simple quickdraw simple dataset with random mask
    """

    def __init__(self, sketch_sum_path, sketch_offset_path, image_sum_path, mode, task_types, get_type, max_length=250, max_size=[256,256], image_size=224, type_size=3, mask_prob=0.8, limit=1000,
                  stroke_type='stroke-5', input_is_complete=True, mask_task_type='task', normalization_type='var_scale', max_scale_factor=10, max_cls_cache=500, each_max_samples=10000, each_image_max_samples=1000, cls_limit_path=None):
        self.mode = mode
        self.task_types = task_types
        self.get_type = get_type
        self.max_length = max_length
        self.max_size = max_size
        self.image_size = image_size
        self.type_size = type_size
        self.mask_prob = mask_prob
        self.limit = limit
        self.stroke_type = stroke_type
        self.input_is_complete = input_is_complete
        self.max_cls_cache = max_cls_cache
        self.each_max_samples = each_max_samples
        self.each_image_max_samples = each_image_max_samples
        self.mask_task_type = mask_task_type
        self.normalization_type = normalization_type
        self.max_scale_factor = max_scale_factor
        self.cls_in_input = 'sketchclsinput' in self.task_types
        self.rel_in_input = 'sketchretrieval' in self.task_types
        self.stroke_dim = STROKE_DIMS[stroke_type]

        assert self.mask_task_type in ['full', 'task', 'single']
        assert self.normalization_type in ['global_scale','var_scale', 'max_scale']
        #self.each_val_samples = each_val_samples
        self.load_sketch_dataset(sketch_sum_path, sketch_offset_path, cls_limit_path)
        self.load_image_dataset(image_sum_path)
        self._make_transform()

    def __len__(self):
        return self.sketch_num

    '''
    Output:
        mask_input[batch, seq_len+cls_input, input_dim](Masked Input)
        stroke[batch, seq_len, input_dim](Original Input)
        segment[batch, seq_len+cls_in_input]
        length_mask[batch, seq_len+cls_input]
        mask[batch, seq_len+cls_input, input_dim](Random Mask)
        cls[batch] ( Class Target)
    '''

    def __getitem__(self, index):
        return self.get_triplet(index)


    def get_sketch_term(self, index):
        skt_ind = self.index_dict[index]
        path = self.sketch_paths[skt_ind[0]]
        sketches = np.memmap(path, dtype=np.float32, mode='r', shape=(self.offsets[self.mode][self.cls2name[skt_ind[0]]][-1][-1], 3))
        sketch = sketches[skt_ind[1][0]:skt_ind[1][1],:].copy()
        del sketches
        cls = self.sketch_classes[index]
        sketch, segment, length_mask = self.transform_stroke(sketch, self.stroke_type, self.normalization_type,  self.max_length, self.max_size, self.type_size)

        return self.get_prefix_result(sketch, segment, length_mask, cls)
    def get_cate_num(self):
        return {c:len(self.image_cls2index[c]) for c in self.image_cls2index}

    def _make_transform(self):

        size = self.image_size
        self._image_transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
        ])

    def get_image_terms(self, index, cls_index):
        pos_cls, neg_clses = cls_index, [i for i in self.sketch_cls2index if i != cls_index]
        #print(pos_cls, neg_clses)
        neg_cls = np.random.choice(neg_clses, 1)[0]

        pos_index = np.random.choice(self.image_cls2index[pos_cls], 1)[0]
        neg_index = np.random.choice(self.image_cls2index[neg_cls], 1)[0]

        pos_path, neg_path = self.image_paths[pos_index], self.image_paths[neg_index]
        pos_image, neg_image = Image.open(pos_path), Image.open(neg_path)

        return self._image_transform(pos_image), self._image_transform(neg_image)

    def get_prefix_result(self, stroke, segment, length_mask, cls):
        length = int(torch.sum(length_mask))

        if self.cls_in_input:
            cls_input = torch.ones(1, stroke.size(1)).to(dtype=torch.float)
            stroke = torch.cat([(cls_input-2).type(dtype=torch.float), stroke.type(dtype=torch.float)], dim=0)
            segment = torch.cat([(cls_input[0,0:1]-1).type(dtype=torch.long), segment.type(dtype=torch.long)], dim=0)
            length_mask = torch.cat([cls_input[0,0:1].type(dtype=torch.float), length_mask.type(dtype=torch.float)], dim=0)
        if self.rel_in_input:
            cls_input = torch.ones(1, stroke.size(1)).to(dtype=torch.float)
            stroke = torch.cat([(cls_input-3).type(dtype=torch.float), stroke.type(dtype=torch.float)], dim=0)
            segment = torch.cat([(cls_input[0,0:1]-1).type(dtype=torch.long), segment.type(dtype=torch.long)], dim=0)
            length_mask = torch.cat([cls_input[0,0:1].type(dtype=torch.float), length_mask.type(dtype=torch.float)], dim=0)
        # print(stroke.size(), stroke.size(), segment.size(), length_mask.size(), mask.size())
        return stroke, segment, length_mask, cls


    def get_triplet(self, index):
        '''
        SBIR Embedding
        '''
        stroke, segment, length_mask, cls = self.get_sketch_term(index)
        pos_image, neg_image = self.get_image_terms(index, cls)
        # Query sketch, pos image, neg image
        #print(torch.sum(pos_image != 0))
        return stroke, pos_image, neg_image, segment, length_mask,  cls
    def transform_stroke(self, stroke, stroke_type, normalization_type, max_length, max_size, type_size):
        # Insert prefix & Truncating the postions out of max length

        # Generate the length mask, sum it = the length
        length_mask = torch.zeros(max_length)
        length_mask[:len(stroke)] = 1
        '''
        Cls: 0
        Non meaning: 1
        Other Segments: 2-max
        '''
        # Generate the segment
        segment = torch.zeros(len(stroke))
        pre_i, now_i, stroke_i = 0, 0, 2
        for i in range(len(stroke)):
            if stroke[i, 2] == 1:
                now_i = i + 1
                segment[pre_i:now_i] = stroke_i
                stroke_i = stroke_i + 1
                pre_i = now_i
        max_segment = torch.zeros(max_length)
        truncated_length = min(max_length, len(stroke))
        max_segment[:truncated_length] = segment[:truncated_length]
        if len(stroke) < max_length:
            max_segment[len(stroke):max_length] = 1

        segment = max_segment
        # Truncate the stroke
        stroke = stroke[:max_length]
        # Change the stroke dim

        # remove the large gap
        stroke = np.minimum(stroke, self.limit)
        stroke = np.maximum(stroke, -self.limit)

        if normalization_type == 'max_scale' and not stroke_type == 'stroke-discrete':
            bounds = np.array(get_bounds(stroke, factor=1))
            sizes = bounds[[1,3]] - bounds[[0,2]]
            if sizes[0] == 0:
                sizes[0] = 1
            if sizes[1] == 0:
                sizes[1] = 1
            stroke[:,:2] = stroke[:,:2] / sizes.reshape(1,2) * self.max_scale_factor

        elif normalization_type == 'var_scale' and not stroke_type == 'stroke-discrete':
            vars = np.std(stroke[:,:2], axis=0)
            if vars[0] == 0:
                vars[0] = 1
            if vars[1] == 0:
                vars[1] = 1
            means = np.mean(stroke[:,:2], axis=0)
            stroke[:,:2] = (stroke[:,:2] - means.reshape(1,2)) / vars.reshape(1,2)

        if stroke_type == 'stroke-5':
            #print(stroke)
            stroke = to_big_strokes(stroke, max_length)
            #print(stroke)
        elif stroke_type == 'stroke-3':
            stroke = extend_strokes(stroke, max_length)
        elif stroke_type == 'stroke-discrete':
            stroke = to_discrete_strokes(stroke, max_length, max_size)


        return torch.from_numpy(stroke), segment, length_mask

    def load_cls_limit_path(self, cls_limit_path):
        cls_limit_list = []
        if cls_limit_path is not None and cls_limit_path != '':
            with open(cls_limit_path, 'r') as rf:
                for line in rf:
                    cls_limit_list.append(line.strip())
        return cls_limit_list

    def load_image_dataset(self, sum_path):
        self.image_paths = []
        self.image_classes = []
        self.image_cls2index = {}
        ind = 0
        with open(sum_path, 'r') as rf:
            for line in rf:
                image_path, cls_name, mode = line.strip().split('\t')
                if cls_name in self.name2cls:
                    cls_index = self.name2cls[cls_name]
                    if cls_index in self.image_cls2index and len(self.image_cls2index[cls_index]) >= (int(self.mode=='train') * self.each_image_max_samples+int(self.mode=='valid')*self.each_max_samples):
                        continue
                    self.image_paths.append(image_path)
                    self.image_classes.append(cls_index)
                    if cls_index in self.image_cls2index:
                        self.image_cls2index[cls_index].append(ind)
                    else:
                        self.image_cls2index[cls_index] = [ind]
                    ind += 1
        self.image_classes = np.array(self.image_classes)
        self.image_num = len(self.image_classes)

        print('Sketch Num:{}, Image Num:{}, Sketch CLS Num:{}, Image CLS Num:{}'.format(self.sketch_num, self.image_num, len(self.sketch_cls2index), len(self.image_cls2index)))

    def load_sketch_dataset(self, sum_path, offset_path, cls_limit_path):
        self.sketch_paths = []
        self.sketch_classes = []
        self.cls2name = {}
        self.name2cls = {}
        self.index_dict = {}
        self.offsets = pkl.load(open(offset_path, 'rb'))
        self.sketch_cls2index = {}
        self.cls_limit_list = self.load_cls_limit_path(cls_limit_path)
        i, ind = 0, 0
        with open(sum_path, 'r') as rf:
            for line in rf:
                if i >= self.max_cls_cache:
                    break

                # Unpackage
                items = line.strip().split('\t')
                path, num = items[0], int(items[1])
                # Update the dict

                cls_name = get_class(path)
                if len(self.cls_limit_list) > 0 and cls_name not in self.cls_limit_list:
                    continue
                print(path)
                self.cls2name[i] = cls_name
                self.name2cls[cls_name] = i
                self.sketch_paths.append(path.replace('.dat', '_{}.dat'.format(self.mode)))
                for j in range(min(len(self.offsets[self.mode][self.cls2name[i]]), self.each_max_samples)):
                    tmp_offset = self.offsets[self.mode][self.cls2name[i]][j]
                    if tmp_offset[1] - tmp_offset[0] >= self.max_length and self.input_is_complete:
                        # print('skip', self.input_is_complete)
                        continue
                    self.index_dict[ind] = (i, tmp_offset)
                    self.sketch_classes.append(i)
                    if i in self.sketch_cls2index:
                        self.sketch_cls2index[i].append(ind)
                    else:
                        self.sketch_cls2index[i] = [ind]
                    ind += 1
                i += 1

        self.sketch_classes = np.array(self.sketch_classes)
        print('CLS NUM',len(self.sketch_cls2index))
        self.sketch_num = len(self.sketch_classes)

    def get_image_loader(self, **args):
        return DataLoader(ImageDataset(self.image_paths, self.image_classes, self.image_size), **args)
