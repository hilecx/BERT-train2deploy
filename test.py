# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import pickle


if __name__ == "__main__":
    label = [ x for x in self._read_tsv(os.path.join("dat_hscode", "class.tsv"))]


#-----------------------------------------
#hscode分类数据处理 2019/3/12 
#labels: from classify.tsv
class HscodeProcessor(DataProcessor):
  this_data_dir=""
  def get_train_examples(self, data_dir):
    """See base class."""
    self.this_data_dir=data_dir
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""

    """
    if not os.path.exists(os.path.join(FLAGS.output_dir, 'label_list.pkl')):
        with codecs.open(os.path.join(FLAGS.output_dir, 'label_list.pkl'), 'wb') as fd:
            pickle.dump(self.labels, fd)
    """
    # label = [ x for x in self._read_tsv(os.path.join(self.this_data_dir, "class.tsv"))]
    label = []
    for x in self._read_tsv(os.path.join(self.this_data_dir, "class.tsv")):
        print(x)
    return label

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0: 
        continue
      guid = "%s-%s" % (set_type, i)

      #debug (by xmxoxo)
      #print("read line: No.%d" % i)

      text_a = tokenization.convert_to_unicode(line[1])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, label=label))
    return examples
#-----------------------------------------