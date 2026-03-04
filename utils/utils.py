# utils.py

import numpy as np


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = {}
            for key in new_dict.keys():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return {}
        return {key: value.get_value() for key, value in self._dict.items()}


def compute_errors(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 1e-6) & (pred > 1e-6)
    if not np.any(valid):
        nan = float("nan")
        return {
            "a1": nan,
            "a2": nan,
            "a3": nan,
            "abs_rel": nan,
            "rmse": nan,
            "log_10": nan,
            "rmse_log": nan,
            "silog": nan,
            "sq_rel": nan,
        }

    gt = np.clip(gt[valid], 1e-6, None)
    pred = np.clip(pred[valid], 1e-6, None)

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())

    err = np.log(pred) - np.log(gt)
    silog_term = np.mean(err ** 2) - np.mean(err) ** 2
    silog = np.sqrt(max(float(silog_term), 0.0)) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return {
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "abs_rel": abs_rel,
        "rmse": rmse,
        "log_10": log_10,
        "rmse_log": rmse_log,
        "silog": silog,
        "sq_rel": sq_rel,
    }

