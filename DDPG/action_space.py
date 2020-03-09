import numpy as np
import pyflann
import itertools



class Space:
    def __init__(self,
                 low_list,
                 high_list,
                 points_list):
        self.low_list = low_list
        self.high_list = high_list
        self._low = np.array(low_list)
        self._high = np.array(high_list)
        self._points_list = points_list
        self._range = self._high - self._low
        self._n_dimensions = len(self._low)

        self._space = self.init_space()
        self._flann = pyflann.FLANN()

        self.rebuild_flann()

    def rebuild_flann(self):
        self._index = self._flann.build_index(self._space, algorithm='kdtree')

    def init_space(self):
        points_in_each_dim = self._points_list

        axis = list()
        for l,h,pts in zip(self.low_list, self.high_list, points_in_each_dim):
            axis.append(list(np.linspace(l, h, pts)))

        space = list()
        for _ in itertools.product(*axis):
            space.append(list(_))

        return np.array(space)

    def search_point(self,
                     point,
                     k):
        # k number of neighbors to search for
        p_in = self.import_point(point)
        search_res, _ = self._flann.nn_index(p_in, k)
        knns = self._space[search_res]
        p_out = list()
        for p in knns:
            p_out.append(self.export_point(p))

        if k == 1:
            p_out = [p_out]

        return np.array(p_out)

    def import_point(self,
                     point):
        # get point relative to low with scaling from range
        return point

    def export_point(self,
                     point):
        return point

    def shape(self):
        return self._space.shape

    def get_num_actions(self):
        return self.shape()[0]


class Discrete(Space):
    def __init__(self,
                 n):
        super().__init__([0], [n-1], n)

    def export_point(self,
                     point):
        return super().export_point(point).astype(int)
