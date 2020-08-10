import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn')


def plotClassificationReport(report, directory):
    df = pd.DataFrame(report).T

    cr = sns.heatmap(df, annot=True, cmap='coolwarm', cbar=False)
    cr.yaxis.set_ticklabels(cr.yaxis.get_ticklabels(),
                            rotation=0, ha='right', fontsize=10)
    cr.xaxis.set_ticklabels(cr.xaxis.get_ticklabels(),
                            rotation=45, ha='right', fontsize=10)

    plt.savefig(f'{directory}/classification_report.png')
    plt.close()


def plotConfusionMatrix(matrix, class_labels, directory):
    df = pd.DataFrame(matrix, index=class_labels, columns=class_labels)

    hm = sns.heatmap(df, annot=True, cmap='coolwarm', cbar=False)
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(),
                            rotation=0, ha='right', fontsize=10)
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(),
                            rotation=45, ha='right', fontsize=10)

    plt.savefig(f'{directory}/confusion_matrix.png')
    plt.close()


def plotTrainingHistory(history, directory):

    for metric in history.history.keys():
        if not metric.startswith('val_'):

            try:
                plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
            except Exception as e:
                print(f'[ERROR] {e}')

            plt.plot(history.history[metric], label=f'Train {metric}')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.title(metric)
            plt.legend()
            plt.savefig(f'{directory}/training_history_{metric}.png')
            plt.close()


# def plotTrainingHistoryLoss(history, directory):
    # plt.plot(history.history['loss'], label='Train Loss')
    # plt.plot(history.history['val_loss'], label='Val Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss')
    # plt.legend()
    # plt.savefig(f'{directory}/training_history_loss.png')
    # plt.close()


# def plotTrainingHistoryAccuracy(history, directory):
    # plt.plot(history.history['accuracy'], label='Train Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy')
    # plt.legend()
    # plt.savefig(f'{directory}/training_history_accuracy.png')
    # plt.close()


# def plotTrainingHistoryAUC(history, directory):
    # plt.plot(history.history['auc'], label='Train AUC')
    # plt.plot(history.history['val_auc'], label='Val AUC')
    # plt.xlabel('Epoch')
    # plt.ylabel('AUC')
    # plt.title('Area Under ROC')
    # plt.legend()
    # plt.savefig(f'{directory}/training_history_AUC.png')
    # plt.close()


# def plotTrainingHistory(history, directory):
    # print('[INFO] Plotting training history')

    # plotTrainingHistoryAccuracy(history, directory)
    # plotTrainingHistoryLoss(history, directory)
    # plotTrainingHistoryAUC(history, directory)
