from sklearn import metrics
import sklearn.preprocessing
import numpy as np
from scipy.special import expit

class TrueLoss:

    def __init__(self, prediction_model, config):
        self.pred_model= prediction_model
        self.config = config
        self.threshold = 0.0

        self.loss_list = ['auc', 'eer', 'avg_prec', 'mcr', 'precision', 'recall', 'f1', 'mathew', 'jaccard']
        self.loss_list_discrete = ['mcr', 'precision', 'recall', 'f1', 'mathew', 'jaccard']

        self.max_num_thresholds = 100

    # compute the measure
    def compute_session(self, sess, X, Y):

        # compute the estimated target
        Y_hat = sess.run(fetches=self.pred_model.Y_hat,
                         feed_dict={self.pred_model.X: X,
                                    self.pred_model.is_training: False})
        return self.compute(Y_hat=Y_hat, Y=Y)


    # compute the measure
    def compute(self, Y_hat, Y):

        loss = 0

        # the area under the roc curve
        if self.config["loss"] == 'auc':
            fpr, tpr, thresholds = metrics.roc_curve(Y, Y_hat)
            auc = metrics.auc(fpr, tpr)
            loss = 1.0 - auc

        # the equal error rate, i.e. at the threshold where false positive rate equals false negative rate
        elif self.config["loss"] == 'eer':
            fpr, tpr, thresholds = metrics.roc_curve(Y, Y_hat)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
            # take the middle value if there FPR is not exactly same to FNR
            loss = (fpr[eer_idx] + fnr[eer_idx])/2.0

        # the average precision accross all thresholds
        elif self.config["loss"] == 'avg_prec':
            loss = 1.0 - metrics.average_precision_score(y_true=Y, y_score=Y_hat)

        # losses that demand the target to be discrete 0/1
        elif self.config["loss"] in self.loss_list_discrete:

            best_threshold_loss = 1000
            best_threshold_val = 1000

            for threshold in np.percentile(Y_hat, np.arange(0,100,1)):

                Y_binary = Y_hat.copy()
                Y_binary[Y_hat >= threshold] = 1
                Y_binary[Y_hat < threshold] = 0

                if self.config["loss"] == 'mcr':
                    threshold_loss = 1.0 - metrics.accuracy_score(y_true=Y, y_pred=Y_binary)

                elif self.config["loss"] == 'precision':
                    threshold_loss = 1.0 - metrics.precision_score(y_true=Y, y_pred=Y_binary)

                elif self.config["loss"] == 'recall':
                    threshold_loss = 1.0 - metrics.recall_score(y_true=Y, y_pred=Y_binary)

                elif self.config["loss"] == 'f1':
                    threshold_loss = 1.0 - metrics.f1_score(y_true=Y, y_pred=Y_binary)

                elif self.config["loss"] == 'mathew':
                    threshold_loss = 1.0 - (sklearn.metrics.matthews_corrcoef(y_true=Y, y_pred=Y_binary)+1.0)/2.0

                elif self.config["loss"] == 'jaccard':
                    threshold_loss = 1.0 - metrics.jaccard_similarity_score(y_true=Y, y_pred=Y_binary)

                #print(i, threshold, threshold_loss)

                if threshold_loss < best_threshold_loss:
                    best_threshold_loss = threshold_loss
                    best_threshold_val = threshold

            loss = best_threshold_loss

        return loss

