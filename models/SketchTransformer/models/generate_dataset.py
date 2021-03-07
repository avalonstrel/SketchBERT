import numpy as np
import os
import ndjson
# dirpath = '/home/lhy/datasets/QuickDraw/sketchrnn'
# out_dirpath = '/home/lhy/datasets/QuickDraw/sketchrnn_single'
# sum_path = '/home/lhy/datasets/QuickDraw/sketchrnn_sum.txt'
# # for filename in os.listdir(dirpath):
# #     if not filename[0] == '.':
# #         classname = filename[:-4]
# #         data = np.load(os.path.join(dirpath, filename), encoding='latin1')
# #
# #         for split in (data):
# #             classpath = os.path.join(out_dirpath, classname, split)
# #             if not os.path.exists(classpath):
# #                 os.makedirs(classpath)
# #             for i, item in enumerate(data[split]):
# #                 np.save(os.path.join(classpath, '{}_{}_{}.npy'.format(classname.replace(' ', '_'), split, i)), item)
# #     break
# with open(sum_path, 'w') as wf:
#     for filename in os.listdir(dirpath):
#         if (not filename[0] == '.') and 'full' == filename[-8:-4]:
#
#             data = np.load(os.path.join(dirpath, filename), encoding='latin1')
#             num_train = len(data['train'])
#             num_valid = len(data['valid'])
#             num_test = len(data['test'])
#             wf.write('{}\t{}\t{}\t{}\n'.format(os.path.join(dirpath, filename), num_train, num_valid, num_test))
# def preprocess(data_item):
#     sketch = data_item['drawing']
#     sketch_segement = []
#     for i in range(len(sketch)):
#         stroke = sketch[i]
#         stroke_segment = np.zeros((len(stroke[0]), 3))
#         stroke_segment[:, 0] = np.array(stroke[0])
#         stroke_segment[:, 1] = np.array(stroke[1])
#         stroke_segment[0, 2] = 1
#         stroke_segment[1:(len(stroke)-1), 2] = 2
#         stroke_segment[-1, 2] = 3
#         stroke_segment[:, 2] = stroke_segment[:, 2] + i * 10
#         sketch_segement.append(stroke_segment)
#     sketch_segement = np.concatenate(sketch_segement, axis=0)
#     return sketch_segement
#
# dirpath = '/home/lhy/datasets/QuickDraw/raw'
# out_dirpath = '/home/lhy/datasets/QuickDraw/segment'
# sum_path = '/home/lhy/datasets/QuickDraw/segment_sum.txt'
# short_num = 50000
# train_split = 0.7
# valid_split = 0.1
# test_split = 0.2
# with open(sum_path, 'w') as wf:
#     for filename in os.listdir(dirpath):
#         if not filename[0] == '.':
#             classname = filename[:-7]
#             print(classname)
#             data_list = []
#             with open(os.path.join(dirpath, filename)) as df:
#                 data = ndjson.load(df)
#             for j, data_item in enumerate(data):
#                 if j >= short_num:
#                     break
#                 sketch_item = preprocess(data_item)
#                 data_list.append(sketch_item)
#             #data_list = np.stack(data_list)
#             num_train = int(len(data_list) * train_split)
#             num_val = int(len(data_list) * valid_split)
#             num_test = len(data_list) - num_train - num_val
#             wf.write('{}\t{}\t{}\t{}\n'.format(os.path.join(out_dirpath, '{}.npz'.format(classname)), num_train, num_val, num_test))
#             np.savez(os.path.join(out_dirpath,'{}.npz'.format(classname)), train=data_list[:num_train], val=data_list[num_train:num_train+num_val], test=data_list[num_train+num_val:num_train+num_val+num_test])
