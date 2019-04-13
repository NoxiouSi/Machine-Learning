import numpy as np

from minst import load_mnist
from ecoc import ECOC
from boosting import AdaBoost, OptimalDecisionStump, RandomDecisionStump


class HAARFeature:

    def __init__(self, x, y, w, h, direction):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        if direction not in ('horizontal', 'vertical'):
            raise ValueError
        self.split_direction = direction

    def feature_value(self, cum_x):
        if self.split_direction == 'horizontal':
            v_top = self._black(cum_x, self.x, self.y, self.w, self.h / 2)
            v_bottom = self._black(cum_x, self.x, self.y + self.h / 2, self.w, self.h / 2)
            return v_top - v_bottom
        if self.split_direction == 'vertical':
            v_left = self._black(cum_x, self.x, self.y, self.w / 2, self.h)
            v_right = self._black(cum_x, self.x + self.w / 2, self.y, self.w / 2, self.h)
            return v_left - v_right

    @staticmethod
    def _black(cum_x, x, y, w, h):
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        return cum_x[:, x + w, y + h] - cum_x[:, x + w, y] - cum_x[:, x, y + h] + cum_x[:, x, y]


if __name__ == '__main__':
    print('Load MINST')
    x, y = load_mnist(path='minst')
    x = x > 0
    cum_x = np.cumsum(np.cumsum(x, axis=1), axis=2)
    print('Traning ECOC')

    n, width, height = x.shape
    features = np.zeros((n, 200))
    for i in range(100):
        w = (np.random.randint(3, 10) + 1) * 2
        h = np.random.choice([k for k in range(int(130 / w), int(170 / w), 2)], 1)[0]
        x_pos = np.random.randint(int(width - w))
        y_pos = np.random.randint(int(height - h))

        haar_h = HAARFeature(x_pos, y_pos, w, h, 'horizontal')
        features[:, i] = haar_h.feature_value(cum_x)
        haar_v = HAARFeature(x_pos, y_pos, w, h, 'vertical')
        features[:, 100 + i] = haar_v.feature_value(cum_x)

    ecoc = ECOC(lambda: AdaBoost(200, OptimalDecisionStump), k=50)
    ecoc.fit(features, y)
    pred = ecoc.predict(features)
    print('Training Accuracy', np.equal(pred, y).mean())
