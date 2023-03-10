import numpy as np
from scipy.spatial.distance import cdist
import sklearn.datasets as sk

class Evaluation:

    def __init__(self, X, y):
        self.X = X 
        self.y =  y
        self.batchSimilarity()
    def batchSimilarity(self):
        self.sim_mat =  1 - cdist(self.X, self.X, metric='cosine')

    def evaluate(self):

        # Calculate evaluation metrics
        thresholds = np.arange(0, 1.0, 0.001)
        print(thresholds.shape)
        fm, tpr, acc = self.calculate_roc(thresholds, self.sim_mat, self.y)
        eer = self.calculate_eer(thresholds, self.sim_mat, self.y)
        return fm, tpr, acc, eer #f-measure, true positive rate, accuracy, equal error rate
    
    def calculate_eer(self,thresholds, sims, labels):
        nrof_thresholds = len(thresholds)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        frr_train = np.zeros(nrof_thresholds)
        eer_index = 0
        eer_diff = 100000000
        for threshold_idx, threshold in enumerate(thresholds):
            frr_train[threshold_idx], far_train[threshold_idx] = self.calculate_val_far(threshold, sims, labels)
            if abs(frr_train[threshold_idx] - far_train[threshold_idx]) < eer_diff:
                eer_diff = abs(frr_train[threshold_idx] - far_train[threshold_idx])
                eer_index = threshold_idx

        frr, far = frr_train[eer_index], far_train[eer_index]

        eer = (frr + far) / 2

        return eer

    def calculate_val_far(self,threshold, sims, actual_issame):
        predict_issame = np.greater(sims, threshold)
        true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
        false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        n_same = np.sum(actual_issame)
        n_diff = np.sum(np.logical_not(actual_issame))
        if n_diff == 0:
            n_diff = 1
        if n_same == 0:
            return 0, 0
        val = float(true_accept) / float(n_same)
        frr = 1 - val
        far = float(false_accept) / float(n_diff)
        return frr, far
    def calculate_accuracy(self,threshold, sims, actual_issame):
        predict_issame = np.greater(sims, threshold)
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

        tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)  # recall
        fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
        precision = 0 if (tp + fp == 0) else float(tp) / float(tp + fp)
        fm = 2 * precision * tpr / (precision + tpr + 1e-12)
        acc = float(tp + tn) / (sims.size + 1e-12)
        return tpr, fpr, precision, fm, acc
    
    def calculate_roc(self,thresholds, sims, labels):
        nrof_thresholds = len(thresholds)
        tprs = np.zeros((nrof_thresholds))
        fprs = np.zeros((nrof_thresholds))
        acc_train = np.zeros((nrof_thresholds))
        precisions = np.zeros((nrof_thresholds))
        fms = np.zeros((nrof_thresholds))

        # Find the best threshold for the fold

        for threshold_idx, threshold in enumerate(thresholds):
            tprs[threshold_idx], fprs[threshold_idx], precisions[threshold_idx], fms[threshold_idx], acc_train[
                threshold_idx] = self.calculate_accuracy(threshold, sims, labels)

        bestindex = np.argmax(fms)
        bestfm = fms[bestindex]
        besttpr = tprs[bestindex]
        bestacc = acc_train[bestindex]

        return bestfm, besttpr, bestacc


np.random.seed(42)

# tf.config.run_functions_eagerly(True)

if __name__ == "__main__":
    print("Hello world")

    X, y = sk.make_classification(n_samples=960, n_features=30, n_informative=20, n_redundant=0, n_classes=10, 
                              n_clusters_per_class=1, weights=None, random_state=1)
    print(X.shape)

    metrics =  Evaluation(X,y)
    print(metrics.evaluate()) 
