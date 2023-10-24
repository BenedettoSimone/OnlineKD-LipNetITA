import sys, os
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_PATH)

from helpers import text_to_labels
from videos import Video
from aligns import Align
from threadsafe import threadsafe_generator
from keras import backend as K
import numpy as np
import keras
import pickle
import glob
import multiprocessing

def get_list_safe(l, index, size):
    ret = l[index:index+size]
    while size - len(ret) > 0:
        ret += l[0:size - len(ret)]
    return ret

# datasets/[train|val]/<sid>/<id>/<image>.png
# or datasets/[train|val]/<sid>/<id>.mpg
# datasets/align/<id>.align
class BasicGenerator(keras.callbacks.Callback):
    def __init__(self, dataset_path, minibatch_size, img_c, img_w, img_h, frames_n, absolute_max_string_len=54, **kwargs):
        self.dataset_path   = dataset_path
        self.minibatch_size = minibatch_size
        self.blank_label    = self.get_output_size() - 1
        self.img_c          = img_c
        self.img_w          = img_w
        self.img_h          = img_h
        self.frames_n       = frames_n
        self.absolute_max_string_len = absolute_max_string_len
        self.cur_train_index = multiprocessing.Value('i', 0)    # Data can be stored in a shared memory using Value
        self.cur_val_index   = multiprocessing.Value('i', 0)
        self.curriculum      = kwargs.get('curriculum', None)
        self.random_seed     = kwargs.get('random_seed', 13)
        self.vtype               = kwargs.get('vtype', 'mouth')
        self.face_predictor_path = kwargs.get('face_predictor_path', None)
        self.steps_per_epoch     = kwargs.get('steps_per_epoch', None)
        self.validation_steps    = kwargs.get('validation_steps', None)
        # Process epoch is used by non-training generator (e.g: validation)
        # because each epoch uses different validation data enqueuer
        # Will be updated on epoch_begin
        self.process_epoch       = -1
        # Maintain separate process train epoch because fit_generator only use
        # one enqueuer for the entire training, training enqueuer can contain
        # max_q_size batch data ahead than the current batch data which might be
        # in the epoch different with current actual epoch
        # Will be updated on next_train()
        self.shared_train_epoch  = multiprocessing.Value('i', -1)
        self.process_train_epoch = -1
        self.process_train_index = -1
        self.process_val_index   = -1

    def build(self, **kwargs):
        self.train_path     = os.path.join(self.dataset_path, 'train')
        self.val_path       = os.path.join(self.dataset_path, 'val')
        self.align_path     = os.path.join(self.dataset_path, 'align')

        self.build_dataset()
        # Set steps to dataset size if not set
        self.steps_per_epoch  = self.default_training_steps if self.steps_per_epoch is None else self.steps_per_epoch
        self.validation_steps = self.default_validation_steps if self.validation_steps is None else self.validation_steps
        return self

    @property
    def training_size(self):
        return len(self.train_list)

    @property
    def default_training_steps(self):
        return self.training_size / self.minibatch_size

    @property
    def validation_size(self):
        return len(self.val_list)

    @property
    def default_validation_steps(self):
        return self.validation_size / self.minibatch_size

    def get_output_size(self):
        return 28

    def get_cache_path(self):
        return self.dataset_path.rstrip('/') + '.cache'

    def enumerate_videos(self, path):
        video_list = []
        for video_path in glob.glob(path):
            try:
                if os.path.isfile(video_path):
                    video = Video(self.vtype, self.face_predictor_path).from_video(video_path)
                else:
                    video = Video(self.vtype, self.face_predictor_path).from_frames(video_path)
            except AttributeError as err:
                raise err
            except:
                print("Error loading video: "+video_path)
                continue
            if K.image_data_format() == 'channels_first' and video.data.shape != (self.img_c,self.frames_n,self.img_w,self.img_h):
                print("Video "+video_path+" has incorrect shape "+str(video.data.shape)+", must be "+str((self.img_c,self.frames_n,self.img_w,self.img_h))+"")
                continue
            if K.image_data_format() != 'channels_first' and video.data.shape != (self.frames_n,self.img_w,self.img_h,self.img_c):
                print("Video "+video_path+" has incorrect shape "+str(video.data.shape)+", must be "+str((self.frames_n,self.img_w,self.img_h,self.img_c))+"")
                continue
            video_list.append(video_path)
        return video_list

    def enumerate_align_hash(self, video_list):
        align_hash = {}
        for video_path in video_list:
            #get folder name
            folder_name = os.path.splitext(video_path)[0].split(os.path.sep)[-2]

            #get video name
            video_id = os.path.splitext(video_path)[0].split(os.path.sep)[-1]
            video_id=folder_name+"/"+video_id
            align_path = os.path.join(self.align_path, video_id)+".align"

            align_hash[video_id] = Align(self.absolute_max_string_len, text_to_labels).from_file(align_path)
        return align_hash

    def build_dataset(self):
        print("\nEnumerating dataset list from disk...")
        self.train_list = self.enumerate_videos(os.path.join(self.train_path, '*', '*'))
        self.val_list   = self.enumerate_videos(os.path.join(self.val_path, '*', '*'))
        self.align_hash = self.enumerate_align_hash(self.train_list + self.val_list)
        #with open(self.get_cache_path(), 'wb') as fp:
        #    pickle.dump((self.train_list, self.val_list, self.align_hash), fp)

        print("Found {} videos for training.".format(self.training_size))
        print("Found {} videos for validation.".format(self.validation_size))
        print("")

        np.random.shuffle(self.train_list)

    def get_align(self, _id):
        return self.align_hash[_id]

    def get_batch(self, index, size, train):
        if train:
            video_list = self.train_list
        else:
            video_list = self.val_list

        X_data_path = get_list_safe(video_list, index, size)
        X_data = []
        Y_data = []
        label_length = []
        input_length = []
        source_str = []
        for path in X_data_path:
            video = Video().from_frames(path)

            folder_align = path.split(os.path.sep)[-2]
            video_id = path.split(os.path.sep)[-1]
            complete_name = folder_align + '/' + video_id
            align = self.get_align(complete_name)

            video_unpadded_length = video.length
            if self.curriculum is not None:
                video, align, video_unpadded_length = self.curriculum.apply(video, align)

            X_data.append(video.data)
            Y_data.append(align.padded_label)
            label_length.append(align.label_length)
            input_length.append(video.length)
            source_str.append(align.sentence)

        source_str = np.array(source_str)
        label_length = np.array(label_length)
        input_length = np.array(input_length)
        Y_data = np.array(Y_data)

        # normalization over training data
        normal_array = np.array(X_data) / 255

        inputs = {'the_input': normal_array,
                  'the_labels': Y_data,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function

        return (inputs, outputs)

    @threadsafe_generator
    def next_train(self):
        r = np.random.RandomState(self.random_seed)
        while 1:
            with self.cur_train_index.get_lock(), self.shared_train_epoch.get_lock():
                cur_train_index = self.cur_train_index.value
                self.cur_train_index.value += self.minibatch_size
                # Shared epoch increment on start or index >= training in epoch
                if cur_train_index >= self.steps_per_epoch * self.minibatch_size:
                    cur_train_index = 0
                    self.shared_train_epoch.value += 1
                    self.cur_train_index.value = self.minibatch_size
                if self.shared_train_epoch.value < 0:
                    self.shared_train_epoch.value += 1
                # Shared index overflow
                if self.cur_train_index.value >= self.training_size:
                    self.cur_train_index.value = self.cur_train_index.value % self.minibatch_size
                # Calculate differences between process and shared epoch
                epoch_differences = self.shared_train_epoch.value - self.process_train_epoch
            if epoch_differences > 0:
                self.process_train_epoch += epoch_differences
                for i in range(epoch_differences):
                    r.shuffle(self.train_list) # Catch up
                #print ("GENERATOR EPOCH {}".format(self.process_train_epoch))
                #print( self.train_list[0])
            #print ("PI: {}, SI: {}, SE: {}".format(cur_train_index, self.cur_train_index.value, self.shared_train_epoch.value))
            if self.curriculum is not None and self.curriculum.epoch != self.process_train_epoch:
                self.update_curriculum(self.process_train_epoch, train=True)
            #print ("Train [{},{}] {}:{}".format(self.process_train_epoch, epoch_differences, cur_train_index,cur_train_index+self.minibatch_size))

            # During the generation of each batch, the curriculum is applied to the video data and alignment
            ret = self.get_batch(cur_train_index, self.minibatch_size, train=True)

            # if epoch_differences > 0:
            #     print "GENERATOR EPOCH {} - {}:{}".format(self.process_train_epoch, cur_train_index, cur_train_index + self.minibatch_size)
            #     print ret[0]['source_str']
            #     print "-------------------"
            yield ret

    @threadsafe_generator
    def next_val(self):
        while 1:
            with self.cur_val_index.get_lock():
                cur_val_index = self.cur_val_index.value
                self.cur_val_index.value += self.minibatch_size
                if self.cur_val_index.value >= self.validation_size:
                    self.cur_val_index.value = self.cur_val_index.value % self.minibatch_size
            if self.curriculum is not None and self.curriculum.epoch != self.process_epoch:
                self.update_curriculum(self.process_epoch, train=False)
            # print "Val [{}] {}:{}".format(self.process_epoch, cur_val_index,cur_val_index+self.minibatch_size)
            ret = self.get_batch(cur_val_index, self.minibatch_size, train=False)
            yield ret

    def on_train_begin(self, logs={}):
        with self.cur_train_index.get_lock():
            self.cur_train_index.value = 0
        with self.cur_val_index.get_lock():
            self.cur_val_index.value = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.process_epoch = epoch

    def update_curriculum(self, epoch, train=True):
        self.curriculum.update(epoch, train=train)
        print("Epoch {}: {}".format(epoch, self.curriculum))