import os

import tensorflow as tf

from official.nlp import bert
import official.nlp.bert.tokenization

from official.nlp.data.classifier_data_lib import InputExample
from official.nlp.data.classifier_data_lib import DataProcessor

import csv

class FileProcessor(DataProcessor):
  """Processor for the CSV data set. Assumess that first line is the header."""
  def __init__(self, labels, data_dir, train_file="train.csv", dev_file="dev.csv", test_file="test.csv", process_text_fn=bert.tokenization.convert_to_unicode):
    super(FileProcessor, self).__init__(process_text_fn)
    self.labels = labels
    self.data_dir = data_dir
    self.train_file, self.dev_file, self.test_file = train_file, dev_file, test_file

  def get_train_examples(self, data_dir = None):
    """See base class."""
    return self._get_examples(data_dir, self.train_file, "train")

  def get_dev_examples(self, data_dir = None):
    """See base class."""
    return self._get_examples(data_dir, self.dev_file, "dev")

  def get_test_examples(self, data_dir = None):
    """See base class."""
    return self._get_examples(data_dir, self.test_file, "test")

  def get_labels(self):
    """See base class."""
    return self.labels
  
  def _get_examples(self, data_dir, filename, set_type):
    data_dir = self.data_dir if data_dir is None else data_dir
    lines = self._read_csv(os.path.join(data_dir, filename))
    return self._create_examples(lines[0], lines[1:], set_type)

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "CSV"

  @classmethod
  def _read_csv(cls, input_file, delimiter=",", quote=None):
    """Reads a CSV value file."""
    with tf.io.gfile.GFile(input_file, "r") as f:
      reader = csv.reader(f, delimiter=delimiter, quotechar=quote)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  def _create_examples(self, header, lines, set_type):
    """Creates examples for the training/dev/test sets."""
    header = dict([(col, i) for i,col in enumerate(header)])
    text_a_idx = header['text_a']
    text_b_idx = header['text_b'] if 'text_b' in header else None
    label_idx = header['label'] if 'label' in header else None
    examples = []
    for i, line in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      # extact the columns
      text_a = self.process_text_fn(line[text_a_idx])
      text_b = self.process_text_fn(line[text_b_idx]) if text_b_idx is not None else None
      label = self.process_text_fn(line[label_idx]) if set_type != "test" else self.labels[0]
      # construct an example
      example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
      examples.append(example)
    return examples


class DataFrameProcessor(DataProcessor):
  def __init__(self, labels, train_df=None, dev_df=None, test_df=None, process_text_fn=bert.tokenization.convert_to_unicode):
    super(DataFrameProcessor, self).__init__(process_text_fn)
    self.labels = labels
    self.train_df, self.dev_df, self.test_df = train_df, dev_df, test_df

  def get_train_examples(self, data_dir = None):
    """See base class."""
    return self._create_examples(self.train_df, "train")

  def get_dev_examples(self, data_dir = None):
    """See base class."""
    return self._create_examples(self.dev_df, "dev")

  def get_test_examples(self, data_dir = None):
    """See base class."""
    return self._create_examples(self.test_df, "test")

  def get_labels(self):
    """See base class."""
    return self.labels
  
  @staticmethod
  def get_processor_name():
    """See base class."""
    return "DataFrame"

  def _create_examples(self, df, set_type):
    """Creates examples for the training/dev/test sets."""
    if df is None: return []
    examples = []
    for i, line in df.iterrows():
      guid = "%s-%s" % (set_type, i)
      # extact the columns 
      text_a = self.process_text_fn(line['text_a'])
      text_b = self.process_text_fn(line['text_b']) if 'text_b' in df.columns else None
      label = self.process_text_fn(line['label']) if set_type != "test" else self.labels[0]
      # construct an example
      example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
      examples.append(example)
    return examples