from base_nn import BaseNN
from utils import load_embeddings
import tensorflow as tf
import os


class ConvNet(BaseNN):
    def construct(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.parameters['max_opcodes']], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.parameters['num_classes']], name="input_y")
        self.dropout_hidden_keep_prob = tf.placeholder(tf.float32, name="dropout_hidden_keep_prob")

        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.parameters['l2_reg_lambda'], scope=None)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding = tf.Variable(
                tf.random_uniform([len(self.vocabulary_dict), self.parameters['embedding_size']], -1.0, 1.0),
                name="W")
            self.embedded_input = tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.embedded_input_expanded = tf.expand_dims(self.embedded_input, -1)


        pooled_outputs = []
        for k in self.parameters['kernel_sizes']:
            with tf.name_scope("conv-{}x{}".format(k, self.parameters['num_filters'])):
                conv = tf.contrib.keras.layers.Conv2D(filters=self.parameters['num_filters'],
                                                        kernel_size=[k,
                                                                     self.parameters['embedding_size']],
                                                        data_format='channels_last',
                                                        use_bias=True,
                                                        activation="elu")(self.embedded_input_expanded)
                pool = tf.nn.max_pool(conv, [1, conv.shape[1], 1, 1], [1, 1, 1, 1], "VALID")
                pooled_outputs.append(pool)

        with tf.device('gpu:0'), tf.name_scope("features"):
            # Combine all the pooled features
            num_filters_total = self.parameters['num_filters'] * len(self.parameters['kernel_sizes'])
            self.h_pool = tf.concat(axis=3, values=pooled_outputs)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total], name="features")

        with tf.device('/gpu:0'), tf.name_scope("output"):
            self.dropout_output = tf.layers.dropout(self.h_pool_flat, rate=self.dropout_hidden_keep_prob)
            self.scores = tf.layers.dense(self.dropout_output, self.parameters['num_classes'])
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.softmax_probabilities = tf.nn.softmax(self.scores, name="probabilities")

        with tf.name_scope("loss_and_accuracy"):

            unregularized_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y),
                name="unregularized_loss")

            weights = tf.trainable_variables()  # all vars of your graph
            regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)

            loss = tf.add(unregularized_loss, regularization_penalty, name="loss")
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            # Training procedure
        with tf.name_scope("train"):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(self.parameters['learning_rate'], name="Adam")
            grads_and_vars = optimizer.compute_gradients(loss)

            if self.parameters['gradient_clipping'] == True:
                capped_gvs = [
                    (tf.clip_by_value(grad, self.parameters['min_gradient'], self.parameters['max_gradient']), var)
                    for grad, var in
                    grads_and_vars]
                train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step,
                                                     name="train_op")
            else:
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step,
                                                     name="train_op")

        with tf.name_scope("summaries"):
            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries, name="grad_summaries_merged")

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", loss)
            acc_summary = tf.summary.scalar("accuracy", accuracy)

            train_summary_op = tf.summary.merge([loss_summary, acc_summary],
                                                name="train_summary_op")
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary],
                                              name="dev_summary_op")



if __name__ == "__main__":
    model = ConvNet("parameters/parameters_cnn_2x64_3x64_4x64_5x64_6x64_7x64.json")
    model.init_directories("ConvNet_2x64_3x64_4x64_5x64_6x64_7x64")
    model.init_graph()
    #model.load_pretrained_embeddings("/path/to/pretrained/embeddings")


    tfrecords_filepath = "/path/to/tfrecords/"
    model.train([tfrecords_filepath + "training0.tfrecords",
                 tfrecords_filepath + "training1.tfrecords",
                 tfrecords_filepath + "training2.tfrecords",
                 tfrecords_filepath + "training3.tfrecords",
                 tfrecords_filepath + "training4.tfrecords",
                 tfrecords_filepath + "training5.tfrecords",
                 tfrecords_filepath + "training6.tfrecords",
                 tfrecords_filepath + "training7.tfrecords",
                 tfrecords_filepath + "training8.tfrecords"
                 ],
                [tfrecords_filepath + "training9.tfrecords"])