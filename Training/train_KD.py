import sys, os

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_PATH)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# TODO use fstring instead of format

from lipnet.generators import BasicGenerator
from lipnet.callbacks import Statistics, Visualize
from lipnet.curriculums import Curriculum
from lipnet.decoders import Decoder
from lipnet.helpers import labels_to_text
from lipnet.spell import Spell
from lipnet.model import LipNet
import numpy as np
import csv
import datetime
import tensorflow as tf
from lipnet.online_kd import extract_features, ensembling_strategy, compute_ensemble_output, multiloss_function


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
          num_samples_stats, peer_networks_n):
    curriculum = Curriculum(curriculum_rules)
    lip_gen = BasicGenerator(dataset_path=DATASET_DIR,
                             minibatch_size=minibatch_size,
                             img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                             absolute_max_string_len=absolute_max_string_len,
                             curriculum=curriculum, start_epoch=start_epoch, is_val=True).build()

    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


    peer_networks_list = []

    for n in range(peer_networks_n):
        lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                        absolute_max_string_len=absolute_max_string_len, output_size=lip_gen.get_output_size())
        lipnet.summary()

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

    callback_list = []
    for n, lipnet in enumerate(peer_networks_list):
        # define callbacks
        statistics = Statistics(lipnet, lip_gen.next_val(), decoder, num_samples_stats,
                                output_dir=os.path.join(OUTPUT_DIR, run_name, "s{}".format(n)))
        visualize = Visualize(os.path.join(OUTPUT_DIR, run_name, "s{}".format(n)), lipnet, lip_gen.next_val(), decoder,
                              num_display_sentences=minibatch_size)

        statistics.on_train_begin()

        callback_list.append((statistics, visualize))

    student_losses_header = [f"Loss student {n}" for n in range(peer_networks_n)]
    header_losses_csv = ["Epoch - Batch"] + student_losses_header + ["Ensemble loss", "Multiloss value"]

    with open(os.path.join(OUTPUT_DIR, run_name, 'training_metrics.csv'), 'w') as csvfile:
        csvw = csv.writer(csvfile)
        csvw.writerow(header_losses_csv)

    # Training
    for epoch in range(start_epoch, stop_epoch):

        print("Epoch {}/{}".format(epoch + 1, stop_epoch))

        student_predictions = []
        student_losses = []

        # For each batch train simultaneously n students
        # TODO set int(lip_gen.default_training_steps)
        for batch in range(1):
            print("Batch {}/{}".format(batch, int(lip_gen.default_training_steps)))

            x_train, y_train = next(lip_gen.next_train())

            f1_array = np.array([])
            f2_array = np.array([])
            f3_array = np.array([])

            # train all students on the same batch
            with tf.GradientTape(persistent=True) as tape:
                for n, lipnet in enumerate(peer_networks_list):

                    statistics = callback_list[n][0]

                    # Forward pass
                    # this result is a set of prob + losses
                    y_pred = lipnet.model(x_train, training=True)

                    # Compute the mean CTC loss
                    student_ctc_loss = tf.reduce_mean(y_pred[1], axis=0)
                    print("Student {} mean loss: {}".format(n, student_ctc_loss))

                    student_predictions.append((n, y_pred[0]))
                    student_losses.append(student_ctc_loss)

                    # Extract_features
                    f1_array, f2_array, f3_array = extract_features(statistics, x_train, f1_array, f2_array, f3_array)

                # At the end of each batch compute ensemble weights
                student_weights = ensembling_strategy(f1_array, f2_array, f3_array, peer_networks_n)
    
                # Sum weighted predictions to compute ensemble output
                # TODO use logits ?
                ensemble_output = compute_ensemble_output(student_predictions, student_weights)

                # TODO set temperature and distillation_strength
                ensemble_loss, multiloss_value = multiloss_function(peer_networks_n, ensemble_output, x_train, student_predictions, 1,
                                       student_losses, 1)

                print("Multiloss value:  {}".format(multiloss_value))

            # Optimize students
            for n, lipnet in enumerate(peer_networks_list):
                print("Optimizing student {}".format(n))
                #gradients = tape.gradient(student_losses[n], lipnet.model.trainable_variables)
                gradients = tape.gradient(multiloss_value, lipnet.model.trainable_variables)
                adam.apply_gradients(zip(gradients, lipnet.model.trainable_variables))

            # Save ctc losses and multiloss
            with open(os.path.join(OUTPUT_DIR, run_name, 'training_metrics.csv'), 'a') as csvfile:
                csvw = csv.writer(csvfile)
                csvw.writerow([f"Epoch {epoch} - Batch {batch}"] + [student_losses[n].numpy()[0] for n in range(peer_networks_n)] + [ensemble_loss.numpy()[0], multiloss_value.numpy()[0]])


        # Save weights for each student every 5 epochs
        #if (epoch - start_epoch) % 5 == 0:

        # Save weights for each student every epoch
        for n, lipnet in enumerate(peer_networks_list):
            lipnet.model.save_weights(
                os.path.join(OUTPUT_DIR, run_name, "s{}".format(n), "weights{:02d}_peer_{:02d}.h5".format(epoch, n)))

        # Save statistics and decoded phrases for each student
        for callbacks in callback_list:
            statistics = callbacks[0]
            visualize = callbacks[1]

            statistics.on_epoch_end(epoch)
            visualize.on_epoch_end(epoch)








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
    # 10th parameter - num_samples_stats (number of samples for statistics evaluation at each epoch)
    # 11th parameter - number of peer network
    train(run_name, 0, 10, 3, 100, 50, 100, 54, 19, 95, 2)
