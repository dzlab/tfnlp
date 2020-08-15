import os

import tensorflow as tf
import tensorflow_hub as hub

from official.modeling import tf_utils
from official.modeling import performance
from official import nlp
from official.nlp import bert

import official.nlp.bert.bert_models
import official.nlp.bert.tokenization
import official.nlp.bert.run_classifier

from official.nlp.bert import configs as bert_configs
from official.nlp import optimization
from official.utils.misc import distribution_utils

from tfnlp.utils import FileProcessor, DataFrameProcessor

class ClassificationModel:
  def __init__(self, strategy, bert_config, bert_classifier, tokenizer, train_args):
    self.strategy = strategy
    self.bert_config = bert_config
    self.bert_classifier = bert_classifier
    self.tokenizer = tokenizer
    self.train_args = train_args

  def train(self, data_path=None, train_df=None, valid_df=None):
    # distribution strategy
    with self.strategy.scope():
      if data_path is not None:
        self.run_train_on_directory(data_path)
      elif train_df is not None:
        self.run_train_on_dataframe(train_df, valid_df)
 
  def run_train_on_directory(self, data_dir):
    train_data_output_path="./train.tf_record"
    eval_data_output_path="./eval.tf_record"
    max_seq_length = self.train_args['max_seq_length']
    # training parameters
    bs = self.train_args['batch_size']
    eval_bs = self.train_args['eval_batch_size']
    epochs = self.train_args['num_train_epochs']
    steps = int(self.train_args['train_data_size'] / self.train_args['batch_size']) + 1
    # 1. Create a parser that will generate trainig/validation/test examples
    processor = FileProcessor(self.train_args['labels'], data_dir)
    # 2. Then apply the transformation to generate new TFRecord files.
    input_meta_data = (
      nlp.data.classifier_data_lib.generate_tf_record_from_data_file(
        processor=processor,
        data_dir=data_dir,
        tokenizer=self.tokenizer,
        train_data_output_path=train_data_output_path,
        eval_data_output_path=eval_data_output_path,
        max_seq_length=max_seq_length
      )
    )
    # 3. Create tf.data input pipelines from those TFRecord files:
    train_ds = bert.run_classifier.get_dataset_fn(train_data_output_path, max_seq_length, bs, is_training=True)()
    valid_ds = bert.run_classifier.get_dataset_fn(eval_data_output_path, max_seq_length, eval_bs, is_training=False)()
    # 4. Train
    self.bert_classifier.fit(train_ds, validation_data=valid_ds, batch_size=bs, epochs=epochs, steps_per_epoch=steps)

  def run_train_on_dataframe(self, train_df, valid_df=None):
    train_data_output_path="./train.tf_record"
    eval_data_output_path="./eval.tf_record"
    max_seq_length = self.train_args['max_seq_length']
    # training parameters
    bs = self.train_args['batch_size']
    eval_bs = self.train_args['eval_batch_size']
    epochs = self.train_args['num_train_epochs']
    steps = int(self.train_args['train_data_size'] / self.train_args['batch_size']) + 1
    # 1. Create a parser that will generate trainig/validation/test examples
    processor = DataFrameProcessor(self.train_args['labels'], train_df, valid_df)
    # 2. Then apply the transformation to generate new TFRecord files.
    input_meta_data = (
      nlp.data.classifier_data_lib.generate_tf_record_from_data_file(
        processor=processor,
        data_dir=None, # not needed in this case as we read from a DataFrame
        tokenizer=self.tokenizer,
        train_data_output_path=train_data_output_path,
        eval_data_output_path=eval_data_output_path,
        max_seq_length=max_seq_length
      )
    )
    # 3. Create tf.data input pipelines from those TFRecord files:
    train_ds = bert.run_classifier.get_dataset_fn(train_data_output_path, max_seq_length, bs, is_training=True)()
    valid_ds = bert.run_classifier.get_dataset_fn(eval_data_output_path, max_seq_length, eval_bs, is_training=False)()
    # 4. Train
    self.bert_classifier.fit(train_ds, validation_data=valid_ds, batch_size=bs, epochs=epochs, steps_per_epoch=steps)

  def evaluate(self, data_path=None, eval_df=None):
    max_seq_length = self.train_args['max_seq_length']
    eval_bs = self.train_args['eval_batch_size']
    test_data_output_path="./test.tf_record"
    if data_path is not None:
      processor = FileProcessor(self.train_args['labels'], data_path)
    elif eval_df is not None:
      # Encode test datasets
      processor = DataFrameProcessor(self.train_args['labels'], test_df=eval_df)
    nlp.data.classifier_data_lib.file_based_convert_examples_to_features(
      processor.get_test_examples(),
      processor.get_labels(),
      max_seq_length,
      self.tokenizer,
      test_data_output_path
      )
    inputs = bert.run_classifier.get_dataset_fn(test_data_output_path, max_seq_length, eval_bs, is_training=False)()
    logits = self.bert_classifier.predict(inputs)
    label_ids = tf.argmax(logits, axis=1).numpy()
    labels = [self.train_args['labels'][idx] for idx in label_ids]
    return labels, label_ids

  def save(self, export_dir='./saved_model'):
    """Export the model."""
    tf.saved_model.save(self.bert_config, export_dir=os.path.join(export_dir, 'config'))
    # TODO use pickle instead
    tf.saved_model.save(self.tokenizer, export_dir=os.path.join(export_dir, 'tokenizer'))
    tf.saved_model.save(self.bert_classifier, export_dir=os.path.join(export_dir, 'classifier'))
    tf.saved_model.save(self.train_args, export_dir=os.path.join(export_dir, 'train_args'))

  @classmethod
  def from_saved(cls, export_dir='./saved_model'):
    """Restore the model."""
    bert_config = tf.saved_model.load(os.path.join(export_dir, 'config'))
    bert_classifier = tf.saved_model.load(os.path.join(export_dir, 'classifier'))
    tokenizer = tf.saved_model.load(os.path.join(export_dir, 'tokenizer'))
    train_args = tf.saved_model.load(os.path.join(export_dir, 'train_args'))
    return cls(bert_config, bert_classifier, tokenizer, train_args)

  @classmethod
  def from_tfhub(cls, hub_module_url, train_args):
    max_seq_length = train_args['max_seq_length']
    num_labels = train_args['num_labels']
    # distribution strategy
    strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=train_args['distribution_strategy'],
      num_gpus=train_args['num_gpus'],
      tpu_address=train_args['tpu']
      )
    
    with strategy.scope():
      # Download model from TF Hub
      bert_encoder = hub.KerasLayer(hub_module_url, trainable=True)
      # Create tokenizer
      vocab_file = bert_encoder.resolved_object.vocab_file.asset_path.numpy()
      do_lower_case = bert_encoder.resolved_object.do_lower_case.numpy()
      tokenizer = bert.tokenization.FullTokenizer(vocab_file, do_lower_case)
      # Create config
      bert_config = bert_configs.BertConfig(vocab_size=len(tokenizer.vocab))
      initializer = tf.keras.initializers.TruncatedNormal(stddev=bert_config.initializer_range)
      # Create head for classification
      input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
      input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
      input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
      pooled_output, _ = bert_encoder([input_word_ids, input_mask, input_type_ids])
      output = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(pooled_output)
      output = tf.keras.layers.Dense(num_labels, kernel_initializer=initializer, name='output')(output)
      inputs = {'input_word_ids': input_word_ids, 'input_mask': input_mask, 'input_type_ids': input_type_ids}
      bert_classifier = tf.keras.Model(inputs=inputs, outputs=output)
      # Compile model
      cls._compile_model(bert_classifier, train_args)
    return cls(strategy, bert_config, bert_classifier, tokenizer, train_args)
    
  @classmethod
  def from_checkpoint(cls, bert_folder, train_args):
    # distribution strategy
    strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=train_args['distribution_strategy'],
      num_gpus=train_args['num_gpus'],
      tpu_address=train_args['tpu']
      )
    #load bert config file
    bert_config_file = os.path.join(bert_folder, "bert_config.json")
    bert_vocab_file = os.path.join(bert_folder, "vocab.txt")
    bert_model_file = os.path.join(bert_folder, 'bert_model.ckpt')
    # read config files
    bert_config = bert.configs.BertConfig.from_json_file(bert_config_file)
    tokenizer = bert.tokenization.FullTokenizer(vocab_file=bert_vocab_file, do_lower_case=train_args['do_lower_case'])

    with strategy.scope():
      bert_classifier, bert_encoder = bert.bert_models.classifier_model(bert_config, num_labels=train_args['num_labels'])
      # restore checkpoint base model
      checkpoint = tf.train.Checkpoint(model=bert_encoder)
      checkpoint.restore(bert_model_file).assert_consumed()
      # compile model
      cls._compile_model(bert_classifier, train_args)
    return cls(strategy, bert_config, bert_classifier, tokenizer, train_args)

  @staticmethod
  def _compile_model(classifier_model, train_args):
    # creates an optimizer with learning rate schedule
    train_data_size = train_args['train_data_size']
    steps_per_epoch = int(train_data_size / train_args['batch_size'])
    num_train_steps = steps_per_epoch * train_args['num_train_epochs']
    warmup_steps = int(train_args['num_train_epochs'] * train_data_size * 0.1 / train_args['batch_size'])
    optimizer = nlp.optimization.create_optimizer(train_args['init_lr'], num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)
    classifier_model.optimizer = performance.configure_optimizer(optimizer, use_float16=train_args['use_float16'], use_graph_rewrite=train_args['use_graph_rewrite'])
    # Create metrics
    metrics = [
      tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)
    ]
    # Create loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Compile the classification model
    classifier_model.compile(optimizer=classifier_model.optimizer, loss=loss, metrics=metrics)
