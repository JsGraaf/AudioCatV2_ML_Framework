from tensorflow import keras
import numpy as np
import tensorflow as tf
from sklearn import metrics


class BestF1OnVal(keras.callbacks.Callback):
    """
    Computes threshold-free diagnostics on the *validation* set each epoch:
      - val_ap                 : Average Precision (PR-AUC)
      - val_f1_max             : max F1 over thresholds
      - val_best_threshold     : threshold achieving F1_max
      - val_precision_at_best  : precision at best F1
      - val_recall_at_best     : recall at best F1
      - val_recall_at_p90      : max recall where precision >= 0.90 (else 0)
      - val_threshold_at_p90   : a threshold that attains that point (if any)

    Works with:
      - binary sigmoid head: y_pred.shape = (N,1) or (N,)
      - 2-class softmax head: y_pred.shape = (N,2) -> uses p1 = y_pred[:,1]
      - labels as scalar {0,1} or one-hot [1,0]/[0,1]
    """

    def __init__(self, val_ds, is_softmax_binary=False, thresholds=None):
        super().__init__()
        self.val_ds = val_ds
        self.is_softmax_binary = is_softmax_binary
        self.thresholds = (
            np.linspace(0.0, 1.0, 1001)
            if thresholds is None
            else np.asarray(thresholds, float)
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Collect validation labels & scores
        y_true_all, y_score_all = [], []
        for xb, yb in self.val_ds:
            y_pred = self.model.predict(xb, verbose=0)
            # labels -> binary
            yb = yb.numpy()
            if yb.ndim == 2 and yb.shape[-1] == 2:
                yb = yb[:, 1]
            else:
                yb = yb.squeeze()
            # scores -> P(class=1)
            if self.is_softmax_binary and y_pred.ndim == 2 and y_pred.shape[-1] == 2:
                p1 = y_pred[:, 1]
            else:
                p1 = y_pred.squeeze()
            y_true_all.append(yb.astype(np.int32))
            y_score_all.append(p1.astype(np.float32))

        y_true = np.concatenate(y_true_all)
        y_score = np.concatenate(y_score_all)

        # Threshold-free metrics (scikit-learn)
        precision, recall, thr = metrics.precision_recall_curve(y_true, y_score)
        ap = metrics.average_precision_score(y_true, y_score)

        # Best F1 across PR curve
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        best_idx = int(np.nanargmax(f1))
        best_f1 = float(f1[best_idx])
        best_p = float(precision[best_idx])
        best_r = float(recall[best_idx])
        # Map PR point index to a threshold index (PR arrays are length = len(thresholds)+1)
        best_th = float(thr[best_idx - 1]) if best_idx > 0 and len(thr) else 0.5

        # Recall at precision >= 0.90
        mask = precision >= 0.90
        if mask.any():
            idx = int(np.argmax(recall[mask]))
            r_at_p90 = float(recall[mask][idx])
            # threshold at that PR point:
            j = np.flatnonzero(mask)[idx]
            th_at_p90 = float(thr[j - 1]) if j > 0 and len(thr) else 0.5
        else:
            r_at_p90, th_at_p90 = 0.0, 1.0  # nothing qualifies

        logs["val_ap"] = ap
        logs["val_f1_max"] = best_f1
        logs["val_best_threshold"] = best_th
        logs["val_precision_at_best"] = best_p
        logs["val_recall_at_best"] = best_r
        logs["val_recall_at_p90"] = r_at_p90
        logs["val_threshold_at_p90"] = th_at_p90

        print(
            f"\n[VAL] AP={ap:.3f} | F1max={best_f1:.3f} @ th={best_th:.3f} "
            f"(P={best_p:.3f}, R={best_r:.3f}) | R@P≥0.90 = {r_at_p90:.3f} "
            f"@ th={th_at_p90:.3f}"
        )


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


class LRTensorBoard(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        print(lr)
        # print(f"Epoch {epoch+1:03d}: learning rate = {lr:.6f}")


class RecallAtP90(keras.metrics.Metric):
    """
    Recall at 90% precision.

    Works with logits or probabilities. For logits, set from_logits=True.
    Handles arbitrary shapes by flattening (assumes binary labels {0,1}).
    """

    def __init__(
        self,
        from_logits: bool = False,
        precision: float = 0.90,
        num_thresholds: int = 200,
        name: str = "recall_at_p90",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.precision = float(precision)
        self.num_thresholds = int(num_thresholds)

        # Use TF's built-in streaming metric under the hood
        self._inner = keras.metrics.RecallAtPrecision(
            precision=self.precision, num_thresholds=self.num_thresholds
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Expect y_true in {0,1}; cast & flatten
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.reshape(y_true, [-1])

        # Convert logits → probs if requested, then flatten
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.reshape(y_pred, [-1])

        return self._inner.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self._inner.result()

    def reset_state(self):
        self._inner.reset_state()

        import tensorflow as tf


class DelayedReduceLROnPlateau(keras.callbacks.Callback):
    """
    Gate a built-in ReduceLROnPlateau so it only starts after `start_epoch`.

    Parameters mirror Keras' ReduceLROnPlateau, plus:
      start_epoch: int, epochs to skip before enabling plateau checks.
    """

    def __init__(
        self,
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1,
        mode="auto",
        min_delta=1e-4,
        cooldown=0,
        min_lr=0.0,
        start_epoch=5,
    ):
        super().__init__()
        self.start_epoch = int(start_epoch)
        self._inner = keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr,
        )

    # Pass-through plumbing so the inner callback is fully initialized
    def set_model(self, model):
        super().set_model(model)
        self._inner.set_model(model)

    def set_params(self, params):
        super().set_params(params)
        self._inner.set_params(params)

    def on_train_begin(self, logs=None):
        self._inner.on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        # Delay until start_epoch
        if (epoch + 1) < self.start_epoch:
            if self._inner.verbose:
                print(
                    f"[DelayedRLoP] Skipping plateau check (epoch {epoch+1}/{self.start_epoch-1})"
                )
            return
        # After the delay, delegate to the inner scheduler
        self._inner.on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        self._inner.on_train_end(logs)
