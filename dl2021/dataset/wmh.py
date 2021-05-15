import numpy as np
from skimage.measure import label

from dpipe.dataset.segmentation import SegmentationFromCSV


class WMH(SegmentationFromCSV):
    def __init__(self, data_path, modalities=('flair',), target='wmh', metadata_rpath='meta.csv', apply_bm=False):
        super().__init__(data_path=data_path,
                         modalities=modalities,
                         target=target,
                         metadata_rpath=metadata_rpath)
        self.n_domains = len(self.df['fold'].unique())
        self.apply_bm = apply_bm

    def load_image(self, i):
        img = np.float32(super().load_image(i)[0])  # 4D -> 3D
        if self.apply_bm:
            bm = self.load_brain_mask(i) > 0.5
            img[~bm] = img.min()
        return img

    def load_segm(self, i):
        return np.float32(super().load_segm(i))  # already 3D

    def load_tumor_centers(self, i):
        labels, n_labels = label(self.load_segm(i) > 0.5, connectivity=3, return_num=True)
        return np.array([np.int32(np.mean(np.argwhere(labels == l), axis=0)) for l in range(1, n_labels + 1)])

    def load_t1_image(self, i):
        # TODO: 2-channels input? (flair, t1) --> probably we don't need t1 at all --> discuss in paper
        img = np.float32(super().load(i, 't1'))
        if self.apply_bm:
            bm = self.load_brain_mask(i) > 0.5
            img[not bm] = img.min()
        return img

    def load_brain_mask(self, i):
        return np.float32(super().load(i, 'brain_mask'))

    def load_shape(self, i):
        return np.int32(np.shape(self.load_segm(i)))

    def load_spacing(self, i):
        voxel_spacing = np.array([self.df['x'].loc[i], self.df['y'].loc[i], self.df['z'].loc[i]])
        return voxel_spacing

    def load_domain_label(self, i):
        domain_id = self.df['fold'].loc[i]
        return np.eye(self.n_domains)[domain_id]  # one-hot-encoded domain

    def load_domain_label_number(self, i):
        return self.df['fold'].loc[i]
