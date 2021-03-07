# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SketchRNN data loading and image manipulation utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import cv2
import numpy as np
import svgwrite
import cairosvg
import torch
from PIL import Image

def get_bounds(data, factor=1):
  """Return bounds of data."""
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0

  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i, 0]) / factor
    y = float(data[i, 1]) / factor
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)

  return (min_x, max_x, min_y, max_y)

def resize_strokes(data, size=128):
  """Return bounds of data."""
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0

  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i, 0])
    y = float(data[i, 1])
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)

  scale_x = ((max_x-min_x)/size)
  scale_y = ((max_y-min_y)/size)
  if scale_x == 0:
    scale_x = 1
  if scale_y == 0:
    scale_y = 1
  data[:,0] = data[:, 0] / scale_x
  data[:,1] = data[:, 1] / scale_y
  return (min_x/scale_x, max_x/scale_x, min_y/scale_y, max_y/scale_y)
'''
change the segments to sequence [segmen_len, each_segment]-> [seq_len]
'''
def segments2sequence(segments, length_masks):
  '''
  segments[batch, seg_len, seq_len, 4](Not +1)
  length_masks[batch, seg_len, seq_len](Not +1)
  '''
  batch_size, max_segment, max_segment_length, input_dim = segments.size()
  seqs = []
  for batch_i in range(batch_size):
    tmp_seq = []
    for seg_i in range(max_segment):
        tmp_seq.append(segments[batch_i,seg_i,length_masks[batch_i,seg_i] == 1 ,:])
    seqs.append(torch.cat(tmp_seq, dim=0))
  return seqs


def rec_incomplete_strokes(data, mask):
  '''
  Process Both stroke-5 or stroke-3
  '''
  new_data = []
  previous_state = False
  offset = np.zeros(2)
  for i in range(data.shape[0]):
    if mask[i,0] == 0:
      if previous_state:
        offset = data[i,:2].copy()
        #print(new_data[-1])
        new_data[-1][:,3] = 1
        new_data[-1][:,2] = 0
      else:
        offset = offset + data[i,:2]
      previous_state = False
    if mask[i,0] == 1:
      if not previous_state:
        tmp_data = data[i].copy()
        tmp_data[:2] = offset + tmp_data[:2]
        tmp_data[2] = 1
        tmp_data[3] = 0
        new_data.append(tmp_data.reshape((1,)+tmp_data.shape))
        previous_state = True
      else:
        new_data.append(data[i].reshape((1,)+data[i].shape))

  return np.concatenate(new_data, axis=0)


def strokes2drawing(data, size=128, svg_filename='tmp.svg'):
  min_x, max_x, min_y, max_y = resize_strokes(data, size=size-10)
  dims = (10 + max_x - min_x, 10 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = 5 - min_x
  abs_y = 5 - min_y

  p = "M%s,%s " % (abs_x, abs_y)
  command = "m"
  for i in range(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])
    y = float(data[i,1])
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  png_filename = svg_filename.replace('.svg', '.png')
  cairosvg.svg2png(url=svg_filename, write_to=png_filename)
  img_np = np.array(Image.open(png_filename).convert("L"))
  img_np = img_np.reshape(img_np.shape+(1,))
  return cv2.resize(img_np, (size, size))



def draw_strokes(data, factor=0.2, svg_filename = '/tmp/sketch_rnn/svg/sample.svg'):
  tf.gfile.MakeDirs(os.path.dirname(svg_filename))
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = 25 - min_x
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)
  command = "m"
  for i in range(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  #display(SVG(dwg.tostring()))

def make_grid_svg(s_list, grid_space=10.0, grid_space_x=16.0):
  def get_start_and_end(x):
    x = np.array(x)
    x = x[:, 0:2]
    x_start = x[0]
    x_end = x.sum(axis=0)
    x = x.cumsum(axis=0)
    x_max = x.max(axis=0)
    x_min = x.min(axis=0)
    center_loc = (x_max+x_min)*0.5
    return x_start-center_loc, x_end
  x_pos = 0.0
  y_pos = 0.0
  result = [[x_pos, y_pos, 1]]
  for sample in s_list:
    s = sample[0]
    grid_loc = sample[1]
    grid_y = grid_loc[0]*grid_space+grid_space*0.5
    grid_x = grid_loc[1]*grid_space_x+grid_space_x*0.5
    start_loc, delta_pos = get_start_and_end(s)

    loc_x = start_loc[0]
    loc_y = start_loc[1]
    new_x_pos = grid_x+loc_x
    new_y_pos = grid_y+loc_y
    result.append([new_x_pos-x_pos, new_y_pos-y_pos, 0])

    result += s.tolist()
    result[-1][2] = 1
    x_pos = new_x_pos+delta_pos[0]
    y_pos = new_y_pos+delta_pos[1]
  return np.array(result)

def slerp(p0, p1, t):
  """Spherical interpolation."""
  omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
  so = np.sin(omega)
  return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def lerp(p0, p1, t):
  """Linear interpolation."""
  return (1.0 - t) * p0 + t * p1


# A note on formats:
# Sketches are encoded as a sequence of strokes. stroke-3 and stroke-5 are
# different stroke encodings.
#   stroke-3 uses 3-tuples, consisting of x-offset, y-offset, and a binary
#       variable which is 1 if the pen is lifted between this position and
#       the next, and 0 otherwise.
#   stroke-5 consists of x-offset, y-offset, and p_1, p_2, p_3, a binary
#   one-hot vector of 3 possible pen states: pen down, pen up, end of sketch.
#   See section 3.1 of https://arxiv.org/abs/1704.03477 for more detail.
# Sketch-RNN takes input in stroke-5 format, with sketches padded to a common
# maximum length and prefixed by the special start token [0, 0, 1, 0, 0]
# The QuickDraw dataset is stored using stroke-3.
def strokes_to_lines(strokes):
  """Convert stroke-3 format to polyline format."""
  x = 0
  y = 0
  lines = []
  line = []
  for i in range(len(strokes)):
    if strokes[i, 2] == 1:
      x += float(strokes[i, 0])
      y += float(strokes[i, 1])
      line.append([x, y])
      lines.append(line)
      line = []
    else:
      x += float(strokes[i, 0])
      y += float(strokes[i, 1])
      line.append([x, y])
  return lines


def lines_to_strokes(lines):
  """Convert polyline format to stroke-3 format."""
  eos = 0
  strokes = [[0, 0, 0]]
  for line in lines:
    linelen = len(line)
    for i in range(linelen):
      eos = 0 if i < linelen - 1 else 1
      strokes.append([line[i][0], line[i][1], eos])
  strokes = np.array(strokes)
  strokes[1:, 0:2] -= strokes[:-1, 0:2]
  return strokes[1:, :]


def augment_strokes(strokes, prob=0.0):
  """Perform data augmentation by randomly dropping out strokes."""
  # drop each point within a line segments with a probability of prob
  # note that the logic in the loop prevents points at the ends to be dropped.
  result = []
  prev_stroke = [0, 0, 1]
  count = 0
  stroke = [0, 0, 1]  # Added to be safe.
  for i in range(len(strokes)):
    candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
    if candidate[2] == 1 or prev_stroke[2] == 1:
      count = 0
    else:
      count += 1
    urnd = np.random.rand()  # uniform random variable
    if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:
      stroke[0] += candidate[0]
      stroke[1] += candidate[1]
    else:
      stroke = candidate
      prev_stroke = stroke
      result.append(stroke)
  return np.array(result)


def scale_bound(stroke, average_dimension=10.0):
  """Scale an entire image to be less than a certain size."""
  # stroke is a numpy array of [dx, dy, pstate], average_dimension is a float.
  # modifies stroke directly.
  bounds = get_bounds(stroke, 1)
  max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
  stroke[:, 0:2] /= (max_dimension / average_dimension)


def to_normal_strokes(big_stroke):
  """Convert from stroke-5 or stroke-4 format (from sketch-rnn paper) back to stroke-3."""
  l = 0
  stroke_len = big_stroke.shape[1]
  if stroke_len == 4:
    l = len(big_stroke)
  else:
    for i in range(len(big_stroke)):
      if big_stroke[i, 4] > 0:
        l = i
        break
    if l == 0:
      l = len(big_stroke)
  result = np.zeros((l, 3))
  result[:, 0:2] = big_stroke[0:l, 0:2]
  result[:, 2] = big_stroke[0:l, 3]
  return result


def clean_strokes(sample_strokes, factor=100):
  """Cut irrelevant end points, scale to pixel space and store as integer."""
  # Useful function for exporting data to .json format.
  copy_stroke = []
  added_final = False
  for j in range(len(sample_strokes)):
    finish_flag = int(sample_strokes[j][4])
    if finish_flag == 0:
      copy_stroke.append([
          int(round(sample_strokes[j][0] * factor)),
          int(round(sample_strokes[j][1] * factor)),
          int(sample_strokes[j][2]),
          int(sample_strokes[j][3]), finish_flag
      ])
    else:
      copy_stroke.append([0, 0, 0, 0, 1])
      added_final = True
      break
  if not added_final:
    copy_stroke.append([0, 0, 0, 0, 1])
  return copy_stroke

def extend_strokes(stroke, max_len=250):
  """Pad stroke-3 format to given length."""
  result = np.zeros((max_len, stroke.shape[1]), dtype=float)
  l = len(stroke)
  assert l <= max_len
  result[:l] = stroke
  return result

def to_big_strokes(stroke, max_len=250):
  """Converts from stroke-3 to stroke-5 format and pads to given length. """
  # (But does not insert special start token).

  result = np.zeros((max_len, 5), dtype=float)
  l = len(stroke)
  assert l <= max_len
  result[0:l, 0:2] = stroke[:, 0:2]
  result[0:l, 3] = stroke[:, 2]
  result[0:l, 2] = 1 - result[0:l, 3]
  result[l:, 4] = 1
  return result

def disc2stroke5(data, max_size):
  max_size = np.array(max_size)
  new_data = np.zeros(data.shape[:-1]+(5,))
  new_data[:,:2] = data[:,:2] - max_size.reshape(1,2)
  #print('d',data[:,2])
  new_data[np.arange(data.shape[0]), (data[:,2]+2).astype(np.int)] = 1
  #print('n',new_data[:,:])
  return new_data

def to_discrete_strokes(stroke, max_len=250, max_size=[256,256]):
  '''Converts from stroke-3 to discrete version stroke-3 format'''
  result = np.zeros((max_len, 3), dtype=float)
  l = len(stroke)
  size = np.array(max_size)
  assert l <= max_len
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0

  abs_x = 0
  abs_y = 0
  for i in range(len(stroke)):
    x = float(stroke[i, 0])
    y = float(stroke[i, 1])
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)
  scale_x = ((max_x-min_x)/size[0])
  scale_y = ((max_y-min_y)/size[1])

  # resize to max_size
  #print(stroke[:,0].shape, result[:l,0].shape, size[0].shape, (stroke[:, 0] / scale_x).shape)
  result[:l,0] = (stroke[:, 0] / scale_x).astype(np.int) + size[0]
  result[:l,1] = (stroke[:, 1] / scale_y).astype(np.int) + size[1]
  result[:l,2] = stroke[:, 2]
  result[l:,2] = 2
  return result


def get_max_len(strokes):
  """Return the maximum length of an array of strokes."""
  max_len = 0
  for stroke in strokes:
    ml = len(stroke)
    if ml > max_len:
      max_len = ml
  return max_len
