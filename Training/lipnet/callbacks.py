"""
This module defines two custom callback classes for evaluating and visualizing model performance during training.
"""

import os
import csv
import numpy as np
import editdistance
import keras
from wer import wer_sentence
from nltk.translate import bleu_score


class Statistics(keras.callbacks.Callback):
    """
    A custom callback class for calculating and logging statistics during model training.

    Args:
        model_container (object): Keras model used for prediction.
        generator (object): Data generator used for model training.
        decoder (object): Decoder used for decoding model predictions.
        num_samples_stats (int): Number of samples for statistics evaluation.
        output_dir (str): Directory where statistics results will be saved (optional).
    """

    def __init__(self, model_container, generator, decoder, num_samples_stats=95, output_dir=None):
        """
        Initializes an instance of Statistics with the specified parameters.
        """
        self.model_container = model_container
        self.output_dir = output_dir
        self.generator = generator
        self.num_samples_stats = num_samples_stats
        self.decoder = decoder
        self.batches_results = []
        if output_dir is not None and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_statistics(self, num):
        """
        Compute CER, WER and BLEU metrics.
        """
        num_left = num
        data = []

        while num_left > 0:
            output_batch = next(self.generator)[0]
            num_proc = min(output_batch['the_input'].shape[0], num_left)
            y_pred = self.model_container.predict(output_batch['the_input'][0:num_proc])
            input_length = output_batch['input_length'][0:num_proc]
            decoded_res = self.decoder.decode(y_pred, input_length)

            for j in range(0, num_proc):
                data.append((decoded_res[j], output_batch['source_str'][j]))

            num_left -= num_proc

        mean_cer, mean_cer_norm = self.get_mean_character_error_rate(data)
        mean_wer, mean_wer_norm = self.get_mean_word_error_rate(data)
        mean_bleu, mean_bleu_norm = self.get_mean_bleu_score(data)

        return {
            'samples': num,
            'cer': (mean_cer, mean_cer_norm),
            'wer': (mean_wer, mean_wer_norm),
            'bleu': (mean_bleu, mean_bleu_norm)
        }

    def get_mean_tuples(self, data, individual_length, func):
        total = 0.0
        total_norm = 0.0
        length = len(data)
        for i in range(0, length):
            val = float(func(data[i][0], data[i][1]))
            total += val
            total_norm += val / individual_length
        return (total / length, total_norm / length)

    def get_mean_character_error_rate(self, data):
        mean_individual_length = np.mean([len(pair[1]) for pair in data])
        return self.get_mean_tuples(data, mean_individual_length, editdistance.eval)

    def get_mean_word_error_rate(self, data):
        mean_individual_length = np.mean([len(pair[1].split()) for pair in data])
        return self.get_mean_tuples(data, mean_individual_length, wer_sentence)

    def get_mean_bleu_score(self, data):
        wrapped_data = [([reference], hypothesis) for reference, hypothesis in data]
        return self.get_mean_tuples(wrapped_data, 1.0, bleu_score.sentence_bleu)

    def on_train_begin(self, logs={}):
        """
        Callback called at the beginning of model training to initialize statistics logging.
        """
        with open(os.path.join(self.output_dir, 'stats.csv'), 'w') as csvfile:
            csvw = csv.writer(csvfile)
            csvw.writerow(
                ["Epoch", "Samples", "Mean CER", "Mean CER (Norm)", "Mean WER", "Mean WER (Norm)", "Mean BLEU",
                 "Mean BLEU (Norm)"])

    def on_batch_end(self, batch, logs=None):
        """
        Callback called at the end of each batch during model training to save batch results for KD attention.
        """

        y_pred = self.model_container.predict(batch['the_input'])
        input_length = batch['input_length']
        decoded_res = self.decoder.decode(y_pred, input_length)

        # save decoded results in dedicated array to compute
        # useful metrics for KD attention
        for j in range(len(decoded_res)):
            self.batches_results.append((decoded_res[j], batch['source_str'][j]))

        # TODO optimize code: array unuseful
        mean_bleu, mean_bleu_norm = self.get_mean_bleu_score(self.batches_results)
        self.batches_results = []

        return y_pred, mean_bleu, mean_bleu_norm

    def on_epoch_end(self, epoch, logs={}):
        """
        Callback called at the end of each epoch during model training to compute and log statistics.
        """
        stats = self.get_statistics(self.num_samples_stats)

        print(('\n\n[Epoch %d] Out of %d samples: [CER: %.3f - %.3f] [WER: %.3f - %.3f] [BLEU: %.3f - %.3f]\n'
               % (epoch, stats['samples'], stats['cer'][0], stats['cer'][1], stats['wer'][0], stats['wer'][1],
                  stats['bleu'][0], stats['bleu'][1])))

        if self.output_dir is not None:
            print(stats)
            with open(os.path.join(self.output_dir, 'stats.csv'), 'a') as csvfile:
                csvw = csv.writer(csvfile)
                csvw.writerow([epoch, stats['samples'],
                               "{0:.5f}".format(stats['cer'][0]), "{0:.5f}".format(stats['cer'][1]),
                               "{0:.5f}".format(stats['wer'][0]), "{0:.5f}".format(stats['wer'][1]),
                               "{0:.5f}".format(stats['bleu'][0]), "{0:.5f}".format(stats['bleu'][1])])


class Visualize(keras.callbacks.Callback):
    """
    A custom callback class for visualizing and saving results during model training.

    Args:
        output_dir (str): Directory where visualized results will be saved.
        model_container (object): Keras model used for prediction.
        generator (object): Data generator used for model training.
        decoder (object): Decoder used for decoding model predictions.
        num_display_sentences (int): Number of sentences to display and save after each epoch.
    """

    def __init__(self, output_dir, model_container, generator, decoder, num_display_sentences=10):
        self.model_container = model_container
        self.output_dir = output_dir
        self.generator = generator
        self.num_display_sentences = num_display_sentences
        self.decoder = decoder
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_epoch_end(self, epoch, logs={}):
        output_batch = next(self.generator)[0]

        y_pred = self.model_container.predict(output_batch['the_input'][0:self.num_display_sentences])
        input_length = output_batch['input_length'][0:self.num_display_sentences]
        res = self.decoder.decode(y_pred, input_length)

        with open(os.path.join(self.output_dir, 'e%02d.csv' % (epoch)), 'w') as csvfile:
            csvw = csv.writer(csvfile)
            csvw.writerow(["Truth", "Decoded"])
            for i in range(self.num_display_sentences):
                csvw.writerow([output_batch['source_str'][i], res[i]])
