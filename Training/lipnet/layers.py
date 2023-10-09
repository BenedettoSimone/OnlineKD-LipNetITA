from keras.layers import Lambda
from keras import backend as K


def ctc_lambda_func(args):
    """
    Compute the Connectionist Temporal Classification (CTC) loss between model predictions and target labels.

    Args:
        args (tuple): A tuple containing four elements:
            - y_pred (Tensor): Model predictions with shape (batch_size, frames, num_classes), where:
                - batch_size is the number of samples in the batch.
                - frames is the maximum length of the temporal sequence.
                - num_classes is the number of output classes (characters).
            - labels (Tensor): Target labels with shape (batch_size, max_label_length), where:
                - batch_size is the number of samples in the batch.
                - max_label_length is the maximum label length.
            - input_length (Tensor): A sequence of input temporal sequence lengths with shape (batch_size, 1), specifying the actual length of each input temporal sequence in the batch.
            - label_length (Tensor): A sequence of target label lengths with shape (batch_size, 1), specifying the actual length of each label in the batch.

    Returns:
        Tensor: The computed CTC losses between model predictions and target labels.

    """
    y_pred, labels, input_length, label_length = args
    # The 2 is critical here since the first couple of outputs from the RNN tend to be garbage:
    # y_pred = y_pred[:, 2:, :]
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# CTC Layer implementation using Lambda layer
# (because Keras doesn't support extra prams on loss function)
def CTC(name, args):
    return Lambda(ctc_lambda_func, output_shape=(1,), name=name)(args)
