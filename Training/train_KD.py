import sys, os

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_PATH)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from keras.optimizers import Adam
from lipnet.generators import BasicGenerator
from lipnet.callbacks import Statistics, Visualize
from lipnet.curriculums import Curriculum
from lipnet.decoders import Decoder
from lipnet.helpers import labels_to_text
from lipnet.spell import Spell
from lipnet.model import LipNet
import numpy as np
import datetime

np.random.seed(55)

DATASET_DIR = os.path.join(CURRENT_PATH, 'datasets')
OUTPUT_DIR = os.path.join(CURRENT_PATH, 'results')
LOG_DIR = os.path.join(CURRENT_PATH, 'logs')

PREDICT_GREEDY = False
PREDICT_BEAM_WIDTH = 200
PREDICT_DICTIONARY = os.path.join(CURRENT_PATH, 'dictionaries', 'phrases.txt')


def curriculum_rules(epoch):
    return {'sentence_length': -1, 'flip_probability': 0.5, 'jitter_probability': 0.05}


def train(run_name, start_epoch, stop_epoch, img_c, img_w, img_h, frames_n, absolute_max_string_len, minibatch_size,
          peer_networks_n):
    curriculum = Curriculum(curriculum_rules)
    lip_gen = BasicGenerator(dataset_path=DATASET_DIR,
                             minibatch_size=minibatch_size,
                             img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                             absolute_max_string_len=absolute_max_string_len,
                             curriculum=curriculum, start_epoch=start_epoch, is_val=True).build()

    peer_networks_list = []
    for n in range(peer_networks_n):
        lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                        absolute_max_string_len=absolute_max_string_len, output_size=lip_gen.get_output_size())
        lipnet.summary()

        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

        # load weights
        if start_epoch == 0:
            start_file_w = os.path.join(OUTPUT_DIR, 'startWeight/unseen-weights178.h5')
            lipnet.model.load_weights(start_file_w)

        # load preexisting trained weights for the model
        if start_epoch > 0:
            weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
            lipnet.model.load_weights(weight_file)

        peer_networks_list.append(lipnet)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])

    # Training
    for epoch in range(start_epoch, stop_epoch):

        students_weights = []
        students_predictions = []

        print("Epoch {}/{}".format(epoch + 1, stop_epoch))

        for n, lipnet in enumerate(peer_networks_list):
            print("Student: {}".format(n))

            # define callbacks
            # TODO set 95
            statistics = Statistics(lipnet, lip_gen.next_val(), decoder, 5,
                                    output_dir=os.path.join(OUTPUT_DIR, run_name))
            visualize = Visualize(os.path.join(OUTPUT_DIR, run_name), lipnet, lip_gen.next_val(), decoder,
                                  num_display_sentences=minibatch_size)

            b = 0
            # int(lip_gen.default_training_steps)
            # TODO set default training steps
            for batch in range(2):
                print("Batch {}/{}".format(b, int(lip_gen.default_training_steps)))
                x_train, y_train = next(lip_gen.next_train())
                loss = lipnet.model.train_on_batch(x_train, y_train)

                # TODO make prediction on validation data
                # make prediction on current batch, decode results and save them to compute at the end of each epoch
                # metrics useful to perform KD attention
                y_pred = statistics.on_batch_end(x_train)

                # Check if there are existing predictions for the current student
                # The result of this step for each student is (number_of_student, array(pred))
                existing_student_index = [i for i, pred in enumerate(students_predictions) if pred[0] == n]
                print(existing_student_index)
                if existing_student_index:
                    # Save predictions to existing student predictions
                    students_predictions[existing_student_index[0]] = (
                        n, np.concatenate((students_predictions[existing_student_index[0]][1], y_pred), axis=0))
                else:
                    students_predictions.append((n, y_pred))

                b = b + 1

            # For each student compute metrics (weights) useful to KD attention
            mean_bleu, mean_bleu_norm = statistics.on_epoch_end(epoch)

            # TODO save weights
            # TODO use callback Visualize
            students_weights.append(mean_bleu_norm)

        # TODO check normalization of weights

        # At the end of each epoch compute ensemble output
        ensemble_predictions = []

        # Get how many predictions
        for i in range(students_predictions[0][1].shape[0]):
            weighted_predictions = []
            for student_idx, pred in students_predictions:
                # Multiply each matrix 100x28 for the student weight
                weighted_predictions.append(pred[i] * students_weights[student_idx])

            # For each sample (i.e. pred1 of s1, pred1 of s2, pred1 of s3)
            # pred * w1 + pred * w2 + pred * w3
            ensemble_prediction = np.sum(weighted_predictions, axis=0)
            ensemble_predictions.append(ensemble_prediction)

        ensemble_predictions = np.array(ensemble_predictions)


if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # 1st parameter - run_name
    # 2nd parameter - start_epoch
    # 3rd parameter - stop_epoch
    # 4th parameter - img_c (num of channel)
    # 5th parameter - img_w
    # 6th parameter - img_h
    # 7th parameter - frames_n
    # 8th parameter - absolute_max_string_length (max len of sentences)
    # 9th parameter - minibatch_size
    # 10th parameter - number of peer network
    train(run_name, 0, 10, 3, 100, 50, 100, 54, 19, 2)
