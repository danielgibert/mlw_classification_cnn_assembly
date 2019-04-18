import tensorflow as tf
import datetime
import json
import time
import os
import logging
import sys
from utils import parse_labels
from utils import parse_inputs
from utils import load_vocabulary
from utils import load_embeddings
from utils import store_metadata
from tensorflow.contrib.tensorboard.plugins import projector


class BaseNN(object):
    def __init__(self, parameters):
        """
        Constructor. It loads the hyperparameters of the network
        Parameters
        ---------
        parameters: str
            File containing the parameters of the network
        """
        self.parameters = self.load_parameters(parameters)
        self.session_conf = tf.ConfigProto(
            allow_soft_placement=self.parameters['allow_soft_placement'],
            log_device_placement=self.parameters['log_device_placement']
        )
        self.vocabulary_dict = load_vocabulary(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), self.parameters['vocabulary']))
        self.inverse_vocabulary_dict = load_vocabulary(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), self.parameters['inverse_vocabulary']))

    def load_parameters(self, parameters_path):
        """
        It loads the network parameters

        Parameters
        ----------
        parameters_path: str
            File containing the parameters of the network
        """
        with open(parameters_path, "r") as param_file:
            params = json.load(param_file)
        return params

    def init_directories(self, out_dir=None):
        """
        Creates the output directory and the folder structure where the model and summaries will be saved

        Parameters
        ----------
        out_dir: str
            Output directory
        """
        timestamp = str(int(time.time()))
        if out_dir is not None:
            self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", out_dir))
        else:
            self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.logger = logging.getLogger('Convolutional Neural Networks for Classification of Malware Assembly Code')
        hdlr = logging.FileHandler(os.path.join(self.out_dir, 'output.log'), mode="a")

        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.INFO)

        # Create summaries directories if not exists
        self.train_summary_dir = os.path.join(self.out_dir, "summaries", "train")
        self.dev_summary_dir = os.path.join(self.out_dir, "summaries", "dev")

        # Summaries base dir
        self.summaries_dir = os.path.abspath(os.path.join(self.out_dir, "summaries"))
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        if not os.path.exists(self.summaries_dir):
            os.makedirs(self.summaries_dir)
        self.early_stopping_dir = os.path.abspath(os.path.join(self.summaries_dir, "early_stopping"))
        self.early_stopping_prefix = os.path.join(self.early_stopping_dir, "model")
        if not os.path.exists(self.early_stopping_dir):
            os.makedirs(self.early_stopping_dir)

    def init_writers(self):
        """
        Initializes train and dev summary writers (Needed for Tensorboard)
        """
        self.train_summary_writer = tf.summary.FileWriter(self.train_summary_dir)
        self.dev_summary_writer = tf.summary.FileWriter(self.dev_summary_dir)

    def init_graph(self):
        """
        Constructs the graph and initializes the session.
        """
        self.network_graph = tf.Graph()
        with self.network_graph.as_default():
            self.construct()
            self.network_session = tf.Session(config=self.session_conf, graph=self.network_graph)
            self.init_writers()
            self.train_summary_writer.add_graph(self.network_graph)
            self.dev_summary_writer.add_graph(self.network_graph)

            store_metadata(self.train_summary_dir + "/metadata.tsv", self.inverse_vocabulary_dict)

            # Configure embeddings_metadata
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = self.network_graph.get_operation_by_name("embedding/W").outputs[0].name
            embedding.metadata_path = os.path.join(self.train_summary_dir, 'metadata.tsv')
            projector.visualize_embeddings(self.train_summary_writer, config)

            self.saver = tf.train.Saver(max_to_keep=self.parameters['max_to_keep'],
                                        keep_checkpoint_every_n_hours=self.parameters[
                                            'keep_checkpoint_every_n_hours'])

            self.network_session.run(tf.global_variables_initializer())
            self.network_session.run(tf.local_variables_initializer())

    def load_model(self, base_path, models_path, retrain=False):
        """
        Restores TensorFlow's model

        Parameters
        ----------
        models_path: str
            Path to the model
        """
        if retrain:
            self.init_directories(base_path)
        self.network_graph = tf.Graph()
        with self.network_graph.as_default():
            self.network_session = tf.Session(config=self.session_conf, graph=self.network_graph)
            with self.network_session.as_default():
                # Load the saved meta graph and restore variables
                self.saver = tf.train.import_meta_graph("{}.meta".format(models_path))
                self.saver.restore(self.network_session, models_path)
                if retrain:
                    self.init_writers()

    def _parse_tfrecord_function(self,
                                 filename):
        features = tf.parse_single_example(
            filename,
            features={
                'mnemonics': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            }
        )
        return features['mnemonics'], features['label']

    def init_datasets_iterators(self, training_filenames, validation_filenames):
        """
        Initializes the input queues for training and validation

        Parameters
        ----------
        training_set: list

        Return
        ------
        tf.Graph:   queues graph
        tf.Session: queues session
        list: training placeholders
        list: validation placeholders

        """
        with self.network_graph.as_default():
            filenames = tf.placeholder(tf.string, shape=[None])

            training_dataset = tf.data.TFRecordDataset(filenames)
            training_dataset = training_dataset.map(self._parse_tfrecord_function)
            training_dataset = training_dataset.shuffle(self.parameters['buffer_size'])
            training_dataset = training_dataset.shuffle(self.parameters['buffer_size'])
            training_dataset = training_dataset.repeat(self.parameters["max_epochs"])
            training_dataset = training_dataset.batch(self.parameters["batch_size"])

            validation_dataset = tf.data.TFRecordDataset(filenames)
            validation_dataset = validation_dataset.map(self._parse_tfrecord_function)
            validation_dataset = validation_dataset.shuffle(self.parameters['buffer_size'])
            validation_dataset = validation_dataset.repeat(self.parameters["max_epochs"])
            validation_dataset = validation_dataset.batch(self.parameters["batch_size"])

            training_iterator = training_dataset.make_initializable_iterator()
            self.network_session.run(training_iterator.initializer, feed_dict={filenames: training_filenames})
            x_train, y_train = training_iterator.get_next()

            validation_iterator = validation_dataset.make_initializable_iterator()
            self.network_session.run(validation_iterator.initializer, feed_dict={filenames: validation_filenames})
            x_dev, y_dev = validation_iterator.get_next()

        return [x_train, y_train], [x_dev, y_dev]

    def train(self, training_filenames, validation_filenames, min_accuracy=None):

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                self.network_graph.get_operation_by_name("input_x").outputs[0]: x_batch,
                self.network_graph.get_operation_by_name("input_y").outputs[0]: y_batch,
                self.network_graph.get_operation_by_name("dropout_hidden_keep_prob").outputs[0]: self.parameters[
                    "dropout_hidden_keep_prob"]
            }


            start_time = time.time()
            _, step, train_summaries, grad_summaries, loss, accuracy = self.network_session.run(
                [self.network_graph.get_operation_by_name("train/train_op").outputs[0],
                 self.network_graph.get_operation_by_name("train/global_step").outputs[0],
                 self.network_graph.get_operation_by_name("summaries/train_summary_op/train_summary_op").outputs[0],
                 self.network_graph.get_operation_by_name(
                     "summaries/grad_summaries_merged/grad_summaries_merged").outputs[0],
                 self.network_graph.get_operation_by_name("loss_and_accuracy/loss").outputs[0],
                 self.network_graph.get_operation_by_name("loss_and_accuracy/accuracy").outputs[0]],
                feed_dict)
            end_time = time.time()
            time_str = datetime.datetime.now().isoformat()
            self.logger.log(logging.INFO, "Step: {} Loss: {} Accuracy: {} - {} seconds/batch".format(step,
                                                                                                     loss,
                                                                                                     accuracy,
                                                                                                     end_time - start_time))
            print(
                "{}: step {}, loss {:g}, acc {:g}, Time: {}".format(time_str, step, loss, accuracy,
                                                                    end_time - start_time))
            self.train_summary_writer.add_summary(train_summaries, step)
            if step % 10 == 0:
                self.train_summary_writer.add_summary(grad_summaries, step)

            return loss, accuracy

        def dev_step(x_batch, y_batch):
            feed_dict = {
                self.network_graph.get_operation_by_name("input_x").outputs[0]: x_batch,
                self.network_graph.get_operation_by_name("input_y").outputs[0]: y_batch,
                self.network_graph.get_operation_by_name("dropout_hidden_keep_prob").outputs[0]: 0.0
            }
            start_time = time.time()
            step, summaries, loss, accuracy = self.network_session.run(
                [self.network_graph.get_operation_by_name("train/global_step").outputs[0],
                 self.network_graph.get_operation_by_name("summaries/dev_summary_op/dev_summary_op").outputs[0],
                 self.network_graph.get_operation_by_name("loss_and_accuracy/loss").outputs[0],
                 self.network_graph.get_operation_by_name("loss_and_accuracy/accuracy").outputs[0]],
                feed_dict)
            end_time = time.time()
            time_str = datetime.datetime.now().isoformat()
            self.logger.log(logging.INFO,
                            "Validation Step: {} Loss: {} Accuracy: {} - {} seconds/batch".format(step,
                                                                                                  loss,
                                                                                                  accuracy,
                                                                                                  end_time - start_time))
            print(
                "{}: step {}, loss {:g}, acc {:g} Time: {}".format(time_str, step, loss, accuracy,
                                                                   end_time - start_time))
            self.dev_summary_writer.add_summary(summaries, step)
            return loss, accuracy

        next_element_training, next_element_validation = self.init_datasets_iterators(training_filenames,
                                                                                      validation_filenames)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        current_step = tf.train.global_step(self.network_session,
                                            self.network_graph.get_operation_by_name("train/global_step").outputs[
                                                0])

        try:
            while current_step < self.parameters['max_iterations']:
                x_train, y_train = self.network_session.run(next_element_training)
                x_train = parse_inputs(x_train, self.vocabulary_dict)
                y_train = parse_labels(y_train, self.parameters['num_classes'])

                # Run one step of the model.
                train_loss, train_accuracy = train_step(x_train,
                                                        y_train)

                current_step = tf.train.global_step(self.network_session,
                                                    self.network_graph.get_operation_by_name(
                                                        "train/global_step").outputs[0])

                if current_step % self.parameters['evaluate_every'] == 0:
                    print("\nEvaluation:")
                    x_dev, y_dev = self.network_session.run(next_element_validation)
                    x_dev = parse_inputs(x_dev, self.vocabulary_dict)
                    y_dev = parse_labels(y_dev, self.parameters['num_classes'])

                    dev_loss, dev_accuracy = dev_step(x_dev,
                                                      y_dev)

                    if min_accuracy is not None:
                        if dev_accuracy >= min_accuracy:
                            path = self.saver.save(self.network_session, self.train_summary_dir,
                                                   global_step=current_step)
                            print("Saved model checkpoint to {}\n".format(path))

                if current_step % self.parameters['checkpoint_every'] == 0:
                    path = self.saver.save(self.network_session, self.train_summary_dir,
                                           global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

                if current_step > self.parameters['max_iterations']:
                    print "Why is the fuck entering here!"
                    raise tf.error.OutOfRangeError

        except tf.errors.OutOfRangeError:
            self.logger.log(logging.CRITICAL,
                            'Done training for %d epochs, %d steps.' % (
                                self.parameters['max_epochs'], current_step))
            print('Done training for %d epochs, %d steps.' % (self.parameters['max_epochs'], current_step))
            path = self.saver.save(self.network_session, self.train_summary_dir, global_step=current_step)
            self.logger.log(logging.CRITICAL, "Saved model checkpoint to {}\n".format(path))
            print("Saved model checkpoint to {}\n".format(path))


        finally:
            # When done, ask the threads to stop.
            self.logger.log(logging.CRITICAL, "Finished training.")
            coord.request_stop()

        self.network_session.close()

    def load_pretrained_embeddings(self, embeddings_filepath):
        embeddings_array = load_embeddings(embeddings_filepath, self.vocabulary_dict, embedding_size=self.parameters['embedding_size'])
        self.network_session.run(self.embedding.assign(embeddings_array))

    def predict(self, mnemonics_raw):
        input_x = self.network_graph.get_operation_by_name("input_x").outputs[0]
        dropout_hidden_keep_prob = self.network_graph.get_operation_by_name("dropout_hidden_keep_prob").outputs[0]

        predictions = self.network_graph.get_operation_by_name("output/predictions").outputs[0]
        probabilities = self.network_graph.get_operation_by_name("output/probabilities").outputs[0]

        ypred, probs = self.network_session.run([predictions, probabilities], {input_x: mnemonics_raw,
                                                                               dropout_hidden_keep_prob: 0.0})

        return ypred, probs

    def close_session(self):
        """
        Closes current session
        """
        self.network_session.close()

    def finalize_graph(self):
        """
        Finalizes the graph. Making it read-only
        """
        self.network_graph.finalize()