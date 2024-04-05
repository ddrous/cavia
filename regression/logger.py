import time

import numpy as np


class Logger:

    def __init__(self):
        self.train_loss = []
        self.test_loss = []

        self.adapt_loss = []
        self.adapt_loss_test = []
        self.adapt_losses_test = []

        self.best_valid_model = None

    def print_info(self, iter_idx, start_time):
        # print(
        #     'Iter {:<4} - time: {:<5} - [train] loss: {:<6} (+/-{:<6}) - [valid] loss: {:<6} (+/-{:<6}) - [test] loss: {:<6} (+/-{:<6})'.format(
        #         iter_idx,
        #         int(time.time() - start_time),
        #         np.round(self.train_loss[-1], 4),
        #         np.round(self.train_conf[-1], 4),
        #         np.round(self.valid_loss[-1], 4),
        #         np.round(self.valid_conf[-1], 4),
        #         np.round(self.test_loss[-1], 4),
        #         np.round(self.test_conf[-1], 4),
        #     )
        # , flush=True)

        print(
            'Iter {:<4} - time: {:<5} - [train]: {:<6} - [valid]: {:<6} - [adapt]: {:<6} - [adapt_test]: {:<6}'.format(
                iter_idx,
                int(time.time() - start_time),
                np.round(self.train_loss[-1], 4),
                np.round(self.test_loss[-1], 4),
                np.round(self.adapt_loss[-1], 4),
                np.round(self.adapt_loss_test[-1], 4),
            )
        , flush=True)
        rounded_losses = [np.round(loss, 4) for loss in self.adapt_losses_test[-1]]
        print('Iter {:<4} - All losses: {}'.format(iter_idx, rounded_losses), flush=True)
