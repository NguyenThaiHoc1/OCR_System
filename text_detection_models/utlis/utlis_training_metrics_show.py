import os
import time
import warnings

import pandas as pd
from tensorflow.keras.metrics import Mean


class Metrics(object):
    def __init__(self, names=None, logdir=None, optimizer=None):

        if names is None:
            names = ['loss', ]
        self.names = names
        self.logdir = logdir
        self.optimizer = optimizer

        # parameter metric value
        self.epoch = None
        self.iteration = None
        self.logs = None
        self.history = None

        self.t0 = None
        self.t1 = None
        self.t2 = None
        self.steps = None
        self.steps_val = None

        self.metrics = None
        self.metrics_val = None

        if logdir is not None:
            self.log_path = os.path.join(logdir, 'log.csv')
            self.history_path = os.path.join(logdir, 'history.csv')

            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)

            if os.path.exists(self.log_path):
                os.remove(self.log_path)

            if os.path.exists(self.history_path):
                os.remove(self.history_path)

        self.reset()

    def reset(self):
        self.iteration = 0
        self.epoch = 0

        self.logs = {name: [] for name in self.names}
        self.logs.update({'epoch': [], 'time': []})

        self.history = {name: [] for name in self.names}
        self.history.update({'val_' + n: [] for n in self.names})
        self.history.update({'epoch': [], 'time': []})

        if self.optimizer is not None:
            self.logs.update({'learning_rate': []})
            self.history.update({'learning_rate': []})

    def update(self, values, training=True):
        metric_values = {name: float(value) for name, value in values.items() if value is not None}
        if training:
            self.t2 = time.time()
            self.steps += 1
            for name, value in metric_values.items():
                self.metrics[name].update_state(value)
                self.logs[name].append(value)

            self.logs['epoch'].append(self.epoch)
            self.logs['time'].append(time.time() - self.t0)

            if self.optimizer is not None:
                self.logs['learning_rate'].append(float(self.optimizer.learning_rate))

            # write log to csv
            if self.logdir is not None:
                float_values = {name: value[-1:] for name, value in self.logs.items()}
                dataframe = pd.DataFrame.from_dict(float_values)
                with open(self.log_path, 'a') as f:
                    dataframe.to_csv(f, header=f.tell() == 0, index=False)

        else:
            self.steps_val += 1
            for name, value in metric_values.items():
                self.metrics_val[name].update_state(value)

    def start_epoch(self):

        # setup time for starting of the an epoch
        if self.epoch == 0:
            self.t0 = time.time()  # time to count whole epoch

        self.t1 = time.time()  # time to count each epoch

        # add 1 when we start an epoch
        self.epoch += 1

        # reset all step, it is the process of the dataset's loop
        self.steps = 0
        self.steps_val = 0

        # setting metrics train and vali with TF.metrics
        self.metrics = {name: Mean() for name in self.names}
        self.metrics_val = {name: Mean() for name in self.names}

    def end_epoch(self, verbose=True):
        if self.steps == 0:
            warnings.warn("You don't have anything to show")
            return

        values_metrics = {name: float(value.result()) for name, value in self.metrics.items()}

        if self.steps_val > 0:
            values_metrics.update({'val_' + name: float(value.result()) for name, value in self.metrics_val.items()})

        # adding value history of An Epoch
        for name, value in values_metrics.items():
            self.history[name].append(value)

        self.history['epoch'].append(self.epoch)
        self.history['time'].append(time.time() - self.t0)
        if self.optimizer is not None:
            self.history['learning_rate'].append(float(self.optimizer.learning_rate))

        # Saving history's value in dataframe save to csv
        if self.logdir is not None:
            history_values = {name: value[-1:] for name, value in self.history.items() if len(value) > 0}
            dataframe = pd.DataFrame.from_dict(history_values)
            with open(self.history_path, 'a') as f:
                dataframe.to_csv(f, header=f.tell() == 0, index=False)

        if verbose:
            t1, t2 = self.t1, self.t2
            time_end_epoch = time.time()
            for name, value in self.history.items():
                s = "%s - %5.5f "
                print(s % (name, value[-1]), end='')

            minutes_epoch = (time_end_epoch - t1) / 60
            iteritor_second = self.steps / (t2 - t1)  # each iteritor from Dataset
            string_time = "\n%.1f minutes/epoch - %.2f iter/second" % (minutes_epoch, iteritor_second)
            print(string_time)


class APCalculator(object):

    def __init__(self, overlap=0.5):
        raise NotImplementedError

    def setup_detections(self, gt_boxes, boxes):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError
