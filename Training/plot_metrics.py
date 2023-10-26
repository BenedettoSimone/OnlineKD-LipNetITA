import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_losses(run_name):

    loss_file = pd.read_csv(f'logs/training-{run_name}.csv')

    train_loss = loss_file['ctc_loss']
    val_loss = loss_file['val_ctc_loss']

    plt.plot(train_loss,  color='red')
    plt.plot(val_loss,  color='blue')

    #plt.title('Seventh training')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left')
    #plt.show()
    plt.savefig(f"plots/{run_name}/losses.jpg")
    plt.close()


def plot_val_metrics(run_name):
    metrics_file = pd.read_csv(f'results/{run_name}/stats.csv')

    wer = metrics_file['Mean WER (Norm)']

    plt.plot(wer, color='green')
    #plt.title('Seventh training')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['wer'], loc='upper left')
    #plt.show()
    plt.savefig(f"plots/{run_name}/val_metrics.jpg")
    plt.close()


if __name__ == '__main__':
    run_name = ""

    if not os.path.exists("plots"):
        os.mkdir("plots")

    run_folder = os.path.join("plots", run_name)
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)

    plot_losses(run_name)
    plot_val_metrics(run_name)