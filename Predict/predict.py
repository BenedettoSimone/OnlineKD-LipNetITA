import sys, os
import csv
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
LIPNET_PATH = os.path.join(CURRENT_PATH,'..','Training')
sys.path.insert(0, LIPNET_PATH)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from Training.lipnet.videos import Video
from Training.lipnet.decoders import Decoder
from Training.lipnet.helpers import labels_to_text
from Training.lipnet.spell import Spell
from Training.lipnet.model import LipNet
import tensorflow as tf
import numpy as np

np.random.seed(55)

# single prediction
VIDEO_PATH = os.path.join(CURRENT_PATH,'PredictVideo','2-bs')
WEIGHTS_PATH = os.path.join(CURRENT_PATH,'weights199.h5')


FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH,'..','MouthExtract','shape_predictor_68_face_landmarks.dat')

PREDICT_GREEDY      = False
PREDICT_BEAM_WIDTH  = 200
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','Training','dictionaries','phrases.txt')

def predict(weight_path, video_path, absolute_max_string_len=54, output_size=28):
    print ("\nLoading data from disk...")
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    if os.path.isfile(video_path):
        video.from_video(video_path)
    else:
        video.from_frames(video_path)
    print ("Data loaded.\n")

    if tf.keras.backend.image_data_format() == 'channels_first':
        img_c, frames_n, img_w, img_h = video.data.shape
    else:
        frames_n, img_w, img_h, img_c = video.data.shape

    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=output_size)

    lipnet.model.load_weights(weight_path)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])

    X_data       = np.array([video.data]).astype(np.float32) / 255   # Normalize
    input_length = np.array([len(video.data)])

    # Dummy data
    label_length =  np.full((1, 1), 10)
    Y_data = np.full((1, 54), 10)


    inputs = {'the_input': X_data,
              'the_labels': Y_data,
              'input_length': input_length,
              'label_length': label_length
              }

    logits = lipnet.model(inputs, training=False)[0]
    y_pred = tf.nn.softmax(logits)

    result = decoder.decode(y_pred, input_length)[0]

    return video, result

if __name__ == '__main__':

    #Single prediction
    video, result = predict(WEIGHTS_PATH, VIDEO_PATH)
    print("[ THE PERSON SAID ] > | {} |".format(result))


    '''
     for i in range(0,19):
            VIDEO_PATH = os.path.join(CURRENT_PATH,'PredictVideo',str(i)+'-bs')

            with open('results/'+str(i)+'-bs.csv', 'w', encoding='UTF8') as f:
                header = ['weightName', 'prediction']

                writer = csv.writer(f)

                # write the header
                writer.writerow(header)

                for w in os.listdir():
                    if '.DS_Store' not in w:
                        if '.h5' in w:
                            WEIGHTS_PATH = os.path.join(CURRENT_PATH, w)

                            video, result = predict(WEIGHTS_PATH, VIDEO_PATH)
                            print("[ THE PERSON SAID ] > | {} |".format(result))

                            data = [w, result]

                            # write the data
                            writer.writerow(data)
    '''



