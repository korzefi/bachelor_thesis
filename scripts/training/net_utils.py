# Filip Korzeniewski
# Partially adapted from :
# Yin R, Luusua E, Dabrowski J, Zhang Y, Kwoh CK.
# Tempel: time-series mutation prediction of influenza A viruses via attention-based recurrent neural networks.
# Bioinformatics. 2020 May 1;36(9):2697-2704. doi: 10.1093/bioinformatics/btaa050. PMID: 31999330.

from scripts.training import evaluation
from scripts.utils import get_root_path
from scripts.utils import get_time_string
from scripts.preprocessing.config import CreatingDatasets
from scripts.training.config import ResultsConfig

import matplotlib.pyplot as plt
import torch.nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import time
import math

root_path = get_root_path()


def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def plot_training_history(loss, val_loss, acc, val_acc, mini_batch_scores, mini_batch_labels):
    """
    Plots the loss and accuracy for training and validation over epochs.
    Also plots the logits for a small batch over epochs.
    """
    plt.style.use('ggplot')

    # Plot losses
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(loss, 'b', label='Training')
    plt.plot(val_loss, 'r', label='Validation')
    plt.title('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 3, 2)
    plt.plot(acc, 'b', label='Training')
    plt.plot(val_acc, 'r', label='Validation')
    plt.title('Accuracy')
    plt.legend()

    # Plot prediction dynamics of test mini batch
    plt.subplot(1, 3, 3)
    pos_label, neg_label = False, False
    for i in range(len(mini_batch_labels)):
        if mini_batch_labels[i]:
            score_sequence = [x[i][1] for x in mini_batch_scores]
            if not pos_label:
                plt.plot(score_sequence, 'b', label='Pos')
                pos_label = True
            else:
                plt.plot(score_sequence, 'b')
        else:
            score_sequence = [x[i][0] for x in mini_batch_scores]
            if not neg_label:
                plt.plot(score_sequence, 'r', label='Neg')
                neg_label = True
            else:
                plt.plot(score_sequence, 'r')

    plt.title('Logits')
    plt.legend()
    # plt.savefig('./reports/figures/training_curves.png')
    # plt.savefig(f'{ResultsConfig.RESULTS_DIRPATH}/loss_fig.png', dpi=350)


def plot_attention(weights):
    """
    Plots attention weights in a grid.
    """
    cax = plt.matshow(weights.numpy(), cmap='bone')
    plt.colorbar(cax)
    plt.grid(
        b=False,
        axis='both',
        which='both',
    )
    plt.xlabel('Years')
    plt.ylabel('Examples')
    # plt.savefig('./reports/figures/attention_weights.png')
    # plt.savefig(f'{ResultsConfig.RESULTS_DIRPATH}/weight.png', dpi=350)
    # plt.savefig(f'{ResultsConfig.RESULTS_DIRPATH}/weight.png', dpi=350)


def predictions_from_output(scores):
    """
    Maps logits to class predictions.
    """
    prob = F.softmax(scores, dim=1)
    _, predictions = prob.topk(1)
    return predictions


def calculate_prob(scores):
    """
    Maps logits to class predictions.
    """
    prob = F.softmax(scores, dim=1)
    pred_probe, _ = prob.topk(1)
    return pred_probe


def train_rnn(model, X, Y, X_valid, Y_valid, show_attention):
    """
    Training loop for a model utilizing hidden states.

    verify - enables sanity checks of the model
    epochs - decides the number of training iterations
    learning rate - decides how much the weights are updated each iteration.
    batch_size - decides how many examples are in each mini batch.
    show_attention - decides if attention weights are plotted.
    """

    # training hyperparams
    epochs = model.num_of_epochs
    learning_rate = model.learning_rate
    batch_size = model.batch_size

    print_interval = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = torch.nn.CrossEntropyLoss()
    num_of_examples = X.shape[1]
    num_of_batches = math.floor(num_of_examples / batch_size)

    all_losses = []
    all_val_losses = []
    all_accs = []
    all_pres = []
    all_recs = []
    all_fscores = []
    all_mccs = []
    all_val_accs = []
    best_mcc = 0.0

    # Find mini batch that contains at least one mutation to plot attention
    plot_batch_size = 10
    i = 0
    for j in range(10):
        while Y_valid[i] == 0:
            i += 1

    X_plot_batch = X_valid[:, i:i + plot_batch_size, :]
    Y_plot_batch = Y_valid[i:i + plot_batch_size]
    plot_batch_scores = []

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        running_Ytrue = None
        running_Ypred = None

        hidden = model.init_hidden(batch_size)

        for count in range(0, num_of_examples - batch_size + 1, batch_size):
            repackage_hidden(hidden)

            X_batch = X[:, count:count + batch_size, :]
            Y_batch = Y[count: count + batch_size]

            # training step
            # forward() is called here
            scores, _ = model(X_batch, hidden)
            # logits passed directly,
            # cross-entr func does the same itself what is later done in predictions_from_output func
            loss = criterion(scores, Y_batch)
            # set gradients to 0
            optimizer.zero_grad()
            # compute gradients
            loss.backward()
            # update parameters
            optimizer.step()

            predictions = predictions_from_output(scores)

            running_Ytrue = Y_batch if running_Ytrue is None else torch.cat((running_Ytrue, Y_batch), 0)
            running_Ypred = predictions if running_Ypred is None else torch.cat((running_Ypred, predictions), 0)
            running_loss += loss.item()

        elapsed_time = time.time() - start_time

        epoch_acc, epoch_prec, epoch_rec, epoch_fscore, epoch_mcc = evaluation.evaluate(running_Ytrue, running_Ypred)
        all_accs.append(epoch_acc)
        all_pres.append(epoch_prec)
        all_recs.append(epoch_rec)
        all_fscores.append(epoch_fscore)
        all_mccs.append(epoch_mcc)
        epoch_loss = running_loss / num_of_batches
        all_losses.append(epoch_loss)

        with torch.no_grad():
            model.eval()
            test_scores, _ = model(X_valid, model.init_hidden(Y_valid.shape[0]))
            predictions = predictions_from_output(test_scores)
            predictions = predictions.view_as(Y_valid)
            pred_prob = calculate_prob(test_scores)
            val_acc, val_prec, val_rec, val_fscore, val_mcc = evaluation.evaluate(Y_valid, predictions)

            val_loss = criterion(test_scores, Y_valid).item()
            all_val_accs.append(val_acc)
            all_val_losses.append(val_loss)

            if val_mcc > best_mcc:
                best_mcc = val_mcc
                print('Epoch %d Time %s' % (epoch, get_time_string(elapsed_time)))
                print(
                    f'Best mcc updated: V_loss %.3f\tV_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f' % (
                        val_loss, val_acc, val_prec, val_rec, val_fscore, val_mcc))
                torch.save(model, ResultsConfig.MODEL_PATH)

            plot_scores, _ = model(X_plot_batch, model.init_hidden(Y_plot_batch.shape[0]))
            plot_batch_scores.append(plot_scores)

        if (epoch + 1) % print_interval == 0:
            print('Epoch %d Time %s' % (epoch, get_time_string(elapsed_time)))
            print('T_loss %.3f\tT_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f' % (
                epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_fscore, epoch_mcc))
            print('V_loss %.3f\tV_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f' % (
                val_loss, val_acc, val_prec, val_rec, val_fscore, val_mcc))

    plot_training_history(all_losses, all_val_losses, all_accs, all_val_accs, plot_batch_scores, Y_plot_batch)

    if show_attention:
        with torch.no_grad():
            model.eval()
            _, attn_weights = model(X_plot_batch, model.init_hidden(Y_plot_batch.shape[0]))
            plot_attention(attn_weights)
    plt.show()


def reshape_to_linear(vecs_by_year, window_size=CreatingDatasets.WINDOW_SIZE):
    reshaped = [[]] * len(vecs_by_year[0])

    for year_vecs in vecs_by_year[-window_size:]:
        for i, vec in enumerate(year_vecs):
            reshaped[i] = reshaped[i] + vec.tolist()

    return reshaped


def test_model(model, X_test, Y_test):
    model.eval()
    test_scores, _ = model(X_test, model.init_hidden(Y_test.shape[0]))
    predictions = predictions_from_output(test_scores)
    predictions = predictions.view_as(Y_test)
    pred_prob = calculate_prob(test_scores)
    accuracy, precision, recall, fscore, mcc = evaluation.evaluate(Y_test, predictions)

    print('Test_acc %.3f\tTest_pre %.3f\tTest_rec %.3f\tTest_fscore %.3f\tTest_mcc %.3f' % (
        accuracy, precision, recall, fscore, mcc))


def logistic_regression(X_vecs, Y, X_vecs_valid, Y_valid):
    X = reshape_to_linear(X_vecs)
    X_valid = reshape_to_linear(X_vecs_valid)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, Y)
    train_acc, train_pre, train_rec, train_fscore, train_mcc = evaluation.evaluate(Y_valid, clf.predict(X))

    Y_pred = clf.predict(X_valid)
    val_acc, val_prec, val_rec, val_fscore, val_mcc = evaluation.evaluate(Y_valid, Y_pred)
    print('Logistic regression baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, val_prec, val_rec, val_fscore, val_mcc))
    # roc curve
    y_pred_roc = clf.predict_proba(X_valid)[:, 1]
    fpr_rt_lr, tpr_rt_lr, _ = roc_curve(Y_valid, y_pred_roc)
    print(auc(fpr_rt_lr, tpr_rt_lr))
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_lr, tpr_rt_lr, label='LR')
    plt.legend(loc='best')
