# Filip Korzeniewski
# Partially adapted from :
# Yin R, Luusua E, Dabrowski J, Zhang Y, Kwoh CK.
# Tempel: time-series mutation prediction of influenza A viruses via attention-based recurrent neural networks.
# Bioinformatics. 2020 May 1;36(9):2697-2704. doi: 10.1093/bioinformatics/btaa050. PMID: 31999330.

from scripts.training import validation
from scripts.utils import get_root_path
from scripts.utils import get_time_string
from scripts.preprocessing.config import CreatingDatasets

import matplotlib.pyplot as plt
import torch.nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

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


def verify_model(model, X, Y, batch_size):
    """
    Checks the loss at initialization of the model and asserts that the
    training examples in a batch aren't mixed together by backpropagating.
    """
    print('Sanity checks:')
    criterion = torch.nn.CrossEntropyLoss()
    scores, _ = model(X, model.init_hidden(Y.shape[0]))
    print(' Loss @ init %.3f, expected ~%.3f' % (criterion(scores, Y).item(), -math.log(1 / model.output_dim)))

    mini_batch_X = X[:, :batch_size, :]
    mini_batch_X.requires_grad_()
    criterion = torch.nn.MSELoss()
    scores, _ = model(mini_batch_X, model.init_hidden(batch_size))

    non_zero_idx = 1
    perfect_scores = [[0, 0] for i in range(batch_size)]
    not_perfect_scores = [[1, 1] if i == non_zero_idx else [0, 0] for i in range(batch_size)]

    scores.data = torch.FloatTensor(not_perfect_scores)
    Y_perfect = torch.FloatTensor(perfect_scores)
    loss = criterion(scores, Y_perfect)
    loss.backward()

    zero_tensor = torch.FloatTensor([0] * X.shape[2])
    for i in range(mini_batch_X.shape[0]):
        for j in range(mini_batch_X.shape[1]):
            if sum(mini_batch_X.grad[i, j] != zero_tensor):
                assert j == non_zero_idx, 'Input with loss set to zero has non-zero gradient.'

    mini_batch_X.detach()
    print(' Backpropagated dependencies OK')


def train_rnn(model, verify, X, Y, X_test, Y_test, show_attention):
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

    if verify:
        verify_model(model, X, Y, batch_size)
    all_losses = []
    all_val_losses = []
    all_accs = []
    all_pres = []
    all_recs = []
    all_fscores = []
    all_mccs = []
    all_val_accs = []

    # Find mini batch that contains at least one mutation to plot
    plot_batch_size = 10
    i = 0
    for j in range(10):
        while Y_test[i] == 0:
            i += 1

    X_plot_batch = X_test[:, i:i + plot_batch_size, :]
    Y_plot_batch = Y_test[i:i + plot_batch_size]
    plot_batch_scores = []

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        running_acc = 0
        running_pre = 0
        running_pre_total = 0
        running_rec = 0
        running_rec_total = 0
        epoch_fscore = 0
        running_mcc_numerator = 0
        running_mcc_denominator = 0
        running_rec_total = 0

        hidden = model.init_hidden(batch_size)

        for count in range(0, num_of_examples - batch_size + 1, batch_size):
            repackage_hidden(hidden)

            X_batch = X[:, count:count + batch_size, :]
            Y_batch = Y[count:count + batch_size]

            # training step
            # forward() is called here
            scores, _ = model(X_batch, hidden)
            # logits passed directly, cross-entr func does the same itself what is later done in predictions_from_output func
            loss = criterion(scores, Y_batch)
            # set gradients to 0
            optimizer.zero_grad()
            # compute gradients
            loss.backward()
            # update parameters
            optimizer.step()

            predictions = predictions_from_output(scores)

            conf_matrix = validation.get_confusion_matrix(Y_batch, predictions)
            TP, FP, FN, TN = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
            running_acc += TP + TN
            running_pre += TP
            running_pre_total += TP + FP
            running_rec += TP
            running_rec_total += TP + FN
            running_mcc_numerator += (TP * TN - FP * FN)
            if ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) == 0:
                running_mcc_denominator += 0
            else:
                running_mcc_denominator += math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            running_loss += loss.item()

        elapsed_time = time.time() - start_time
        epoch_acc = running_acc / Y.shape[0]
        all_accs.append(epoch_acc)

        if running_pre_total == 0:
            epoch_pre = 0
        else:
            epoch_pre = running_pre / running_pre_total
        all_pres.append(epoch_pre)

        if running_rec_total == 0:
            epoch_rec = 0
        else:
            epoch_rec = running_rec / running_rec_total
        all_recs.append(epoch_rec)

        if (epoch_pre + epoch_rec) == 0:
            epoch_fscore = 0
        else:
            epoch_fscore = 2 * epoch_pre * epoch_rec / (epoch_pre + epoch_rec)
        all_fscores.append(epoch_fscore)

        if running_mcc_denominator == 0:
            epoch_mcc = 0
        else:
            epoch_mcc = running_mcc_numerator / running_mcc_denominator
        all_mccs.append(epoch_mcc)

        epoch_loss = running_loss / num_of_batches
        all_losses.append(epoch_loss)

        with torch.no_grad():
            model.eval()
            test_scores, _ = model(X_test, model.init_hidden(Y_test.shape[0]))
            predictions = predictions_from_output(test_scores)
            predictions = predictions.view_as(Y_test)
            pred_prob = calculate_prob(test_scores)
            precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, predictions)

            val_loss = criterion(test_scores, Y_test).item()
            all_val_losses.append(val_loss)
            all_val_accs.append(val_acc)

            plot_scores, _ = model(X_plot_batch, model.init_hidden(Y_plot_batch.shape[0]))
            plot_batch_scores.append(plot_scores)

        if (epoch + 1) % print_interval == 0:
            print('Epoch %d Time %s' % (epoch, get_time_string(elapsed_time)))
            print('T_loss %.3f\tT_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f' % (
                epoch_loss, epoch_acc, epoch_pre, epoch_rec, epoch_fscore, epoch_mcc))
            print('V_loss %.3f\tV_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f' % (
                val_loss, val_acc, precision, recall, fscore, mcc))
    plot_training_history(all_losses, all_val_losses, all_accs, all_val_accs, plot_batch_scores, Y_plot_batch)

    # roc curve
    tpr_rnn, fpr_rnn, _ = roc_curve(Y_test, pred_prob)
    print(auc(fpr_rnn, tpr_rnn))
    plt.figure(1)
    # plt.xlim(0, 0.8)
    plt.ylim(0.5, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rnn, tpr_rnn, label=model.__class__.__name__)
    plt.legend(loc='best')

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

def logistic_regression(X_vecs, Y, X_vecs_test, Y_test):
    X = reshape_to_linear(X_vecs)
    X_test = reshape_to_linear(X_vecs_test)
    clf = LogisticRegression(random_state=0).fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, Y_pred)
    print('Logistic regression baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))
    # roc curve
    y_pred_roc = clf.predict_proba(X_test)[:, 1]
    fpr_rt_lr, tpr_rt_lr, _ = roc_curve(Y_test, y_pred_roc)
    print(auc(fpr_rt_lr, tpr_rt_lr))
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_lr, tpr_rt_lr, label='LR')
    plt.legend(loc='best')
