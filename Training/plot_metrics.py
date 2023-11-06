import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_losses(run_name):

    loss_file = pd.read_csv(f'logs/training-{run_name}.csv')

    train_loss = loss_file['ctc_loss']
    val_loss = loss_file['val_ctc_loss']

    plt.plot(train_loss,  color='red')
    plt.plot(val_loss,  color='blue')

    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left')
    #plt.show()
    plt.savefig(f"plots/{run_name}/losses.jpg")
    plt.close()


def plot_val_metrics(run_name):
    metrics_file = pd.read_csv(f'results/{run_name}/stats.csv')

    wer = metrics_file['Mean WER (Norm)']
    cer = metrics_file['Mean CER (Norm)']

    plt.plot(wer, color='green')
    plt.plot(cer, color='orange')

    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['WER', 'CER'], loc='upper left')
    #plt.show()
    plt.savefig(f"plots/{run_name}/val_metrics.jpg")
    plt.close()


def plot_losses_kd(run_name):

    loss_file = pd.read_csv(f'logs/training-{run_name}.csv')

    train_loss_0 = loss_file['train_loss_s0']
    val_loss_0 = loss_file['val_loss_s0']
    train_loss_1 = loss_file['train_loss_s1']
    val_loss_1 = loss_file['val_loss_s1']

    plt.plot(train_loss_0,  color='red')
    plt.plot(val_loss_0,  color='blue')
    plt.plot(train_loss_1, color='green')
    plt.plot(val_loss_1, color='orange')

    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['train_loss_s0', 'val_loss_s0','train_loss_s1', 'val_loss_s1'], loc='upper left')
    #plt.show()
    plt.savefig(f"plots/{run_name}/losses.jpg")
    plt.close()


def plot_val_metrics_kd(run_name):
    metrics_file = pd.read_csv(f'results/{run_name}/s0/stats.csv')
    metrics_file1 = pd.read_csv(f'results/{run_name}/s1/stats.csv')

    wer = metrics_file['Mean WER (Norm)']
    cer = metrics_file['Mean CER (Norm)']
    wer1 = metrics_file1['Mean WER (Norm)']
    cer1 = metrics_file1['Mean CER (Norm)']

    #min_wer_index = wer.idxmin()
    #min_wer_value = wer.min()

    plt.plot(wer, color='green')
    plt.plot(cer, color='orange')
    plt.plot(wer1, color='blue')
    plt.plot(cer1, color='red')

    #plt.grid(True)
    #plt.axhline(min_wer_value, color='green', linestyle='--')
    #plt.axvline(min_wer_index, color='green', linestyle='--')

    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['WER', 'CER', 'WER1', 'CER1'], loc='upper left')
    #plt.show()
    plt.savefig(f"plots/{run_name}/val_metrics.jpg")
    plt.close()


def plot_train_metrics_kd(run_name):
    metrics_file = pd.read_csv(f'results/{run_name}/training_metrics.csv')

    l0 = metrics_file['Loss student 0']
    l1 = metrics_file['Loss student 1']
    el = metrics_file['Ensemble loss']
    ml = metrics_file['Multiloss value']

    plt.plot(l0, color='green')
    plt.plot(l1, color='orange')
    plt.plot(el, color='blue')
    plt.plot(ml, color='red')

    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Loss 0', 'Loss 1', 'Ensemble loss', 'Multiloss'], loc='upper left')
    #plt.show()
    plt.savefig(f"plots/{run_name}/training_metrics.jpg")
    plt.close()


if __name__ == '__main__':
    # Vanilla Training
    #run_name = "2023_10_26_09_22_27"
    #run_name = "2023_10_28_11_30_29"
    #run_name = "2023_10_31_18_23_34"
    #run_name = "2023_11_02_10_55_00"
    run_name = "2023_11_04_16_41_41"


    if not os.path.exists("plots"):
        os.mkdir("plots")

    run_folder = os.path.join("plots", run_name)
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)

    plot_losses(run_name)
    plot_val_metrics(run_name)
    #plot_losses_kd(run_name)
    #plot_val_metrics_kd(run_name)
    #plot_train_metrics_kd(run_name)
