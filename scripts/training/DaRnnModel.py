from scripts.training.config import NetParameters, ResultsConfig
from scripts.training import validation
from scripts.utils import get_root_path
from scripts.utils import get_time_string

import matplotlib.pyplot as plt
import torch.nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

import time
import math

root_path = get_root_path()


class DaRnnModel(torch.nn.Module):
    """
    A Dual-Attention RNN model, attending over both the input at each timestep
    and all hidden states of the encoder to make the final prediction.
    """

    def __init__(self, seq_length, input_dim, output_dim):
        super(DaRnnModel, self).__init__()

        self.n = input_dim
        self.m = NetParameters.hidden_size
        self.T = seq_length
        self.output_dim = output_dim

        self.dropout = torch.nn.Dropout(NetParameters.dropout_p)

        self.encoder = torch.nn.LSTM(self.n, self.m)

        self.We = torch.nn.Linear(2 * self.m, self.T)
        self.Ue = torch.nn.Linear(self.T, self.T)
        self.ve = torch.nn.Linear(self.T, 1)

        self.Ud = torch.nn.Linear(self.m, self.m)
        self.vd = torch.nn.Linear(self.m, 1)
        self.out = torch.nn.Linear(self.m, output_dim)

    def forward(self, x, hidden_state):
        x = self.dropout(x)
        h_seq = []
        for t in range(self.T):
            x_tilde, _ = self.input_attention(x, hidden_state, t)
            ht, hidden_state = self.encoder(x_tilde, hidden_state)
            h_seq.append(ht)

        h = torch.cat(h_seq, dim=0)
        c, beta = self.temporal_attention(h)
        logits = self.out(c)

        return logits, torch.squeeze(beta)

    def input_attention(self, x, hidden_state, t):
        x = x.permute(1, 2, 0)
        h, c = hidden_state
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)
        hc = torch.cat([h, c], dim=2)

        e = self.ve(torch.tanh(self.We(hc) + self.Ue(x)))
        e = torch.squeeze(e)
        alpha = F.softmax(e, dim=1)
        xt = x[:, :, t]

        x_tilde = alpha * xt
        x_tilde = torch.unsqueeze(x_tilde, 0)

        return x_tilde, alpha

    def temporal_attention(self, h):
        h = h.permute(1, 0, 2)
        l = self.vd(torch.tanh((self.Ud(h))))
        l = torch.squeeze(l)
        beta = F.softmax(l, dim=1)
        beta = torch.unsqueeze(beta, 1)
        c = torch.bmm(beta, h)
        c = torch.squeeze(c)

        return c, beta

    def init_hidden(self, batch_size):
        h_init = torch.zeros(1, batch_size, self.m)
        c_init = torch.zeros(1, batch_size, self.m)

        return (h_init, c_init)


def get_confusion_matrix(y_true, y_pred):
    """
    Calculates the confusion matrix from given labels and predictions.
    Expects tensors or numpy arrays of same shape.
    """
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(y_true.shape[0]):
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        elif y_true[i] == 1 and y_pred[i] == 1:
            TP += 1

    conf_matrix = [
        [TP, FP],
        [FN, TN]
    ]

    return conf_matrix


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
    plt.savefig(f'{ResultsConfig.RESULTS_DIRPATH}/loss_fig.png', dpi=350)


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
    plt.savefig(f'{ResultsConfig.RESULTS_DIRPATH}/weight.png', dpi=350)
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


def train_rnn(model, verify, epochs, learning_rate, batch_size, X, Y, X_test, Y_test, show_attention):
    """
    Training loop for a model utilizing hidden states.

    verify - enables sanity checks of the model
    epochs - decides the number of training iterations
    learning rate - decides how much the weights are updated each iteration.
    batch_size - decides how many examples are in each mini batch.
    show_attention - decides if attention weights are plotted.
    """
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
    while not Y_test[i]:
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

            scores, _ = model(X_batch, hidden)
            loss = criterion(scores, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions = predictions_from_output(scores)

            conf_matrix = get_confusion_matrix(Y_batch, predictions)
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
    if epoch + 1 == 50:
        tpr_rnn, fpr_rnn, _ = roc_curve(Y_test, pred_prob)
        print(auc(fpr_rnn, tpr_rnn))
        plt.figure(1)
        # plt.xlim(0, 0.8)
        plt.ylim(0.5, 1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_rnn, tpr_rnn, label='attention')
        plt.legend(loc='best')

    if show_attention:
        with torch.no_grad():
            model.eval()
            _, attn_weights = model(X_plot_batch, model.init_hidden(Y_plot_batch.shape[0]))
            plot_attention(attn_weights)
    plt.show()
