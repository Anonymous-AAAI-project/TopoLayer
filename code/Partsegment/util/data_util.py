import numpy as np
from torch.utils.data import Dataset
import os
import json
import torch
from torch.utils.data import Dataset
import ripser as Rips # calculate PD
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class PDDataStructure():
    def __init__(self, pointcloud, maxdim=2, thresh=0.5, coeff=2, distance_matrix=False, do_cocycles=False,
                 metric="euclidean", n_perm=128):
        ripser = Rips.ripser(pointcloud, maxdim=maxdim, thresh=thresh, coeff=coeff,
                             distance_matrix=distance_matrix, do_cocycles=do_cocycles,
                             metric=metric, n_perm=n_perm)
        # pd_data
        ripser["dgms"][1][np.isinf(ripser["dgms"][1])] = 2
        ripser["dgms"][1] = np.concatenate((ripser["dgms"][1], [[0, 0]]), axis=0)
        ripser["dgms"][2][np.isinf(ripser["dgms"][2])] = 2
        ripser["dgms"][2] = np.concatenate((ripser["dgms"][2], [[0, 0]]), axis=0)

        self.h1_pd_data = torch.tensor(ripser["dgms"][1], dtype=torch.float32)
        self.h2_pd_data = torch.tensor(ripser["dgms"][2], dtype=torch.float32)
        self.pd_data = (self.h1_pd_data, self.h2_pd_data)


# =========== ShapeNet Part =================
class PartNormalDataset(Dataset):
    def __init__(self, npoints=2500, split='train', normalize=False):
        self.npoints = npoints
        self.root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normalize = normalize

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point)) #

            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(self.cat, range(len(self.cat))))
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
        
    def _read_and_process_data(self, file_path):
        with open(file_path, 'r') as f:
            data = np.loadtxt(f).astype(np.float32)
        
        point_set = data[:, 0:3]
        normal = data[:, 3:6]
        seg = data[:, -1].astype(np.int32)

        # PD
        pd_data = PDDataStructure(data).pd_data

        return point_set, normal, seg, pd_data

    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls, pd_data = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            
            point_set, normal, seg, pd_data = self._read_and_process_data(fn[1])

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, cls, pd_data)

        if self.normalize:
            point_set = pc_normalize(point_set)

        choice = np.random.choice(len(seg), self.npoints, replace=True)

        # resample
        # note that the number of points in some points clouds is less than 2048, thus use random.choice
        # remember to use the same seed during train and test for a getting stable result
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice, :]

        return point_set, cls, seg, normal, pd_data

    def __len__(self):
        return len(self.datapath)

class Collatet():
    def __call__(self, batch):
        processed_pointcloud = torch.utils.data.dataloader.default_collate([item[0] for item in batch])
        processed_cls = torch.utils.data.dataloader.default_collate([item[1] for item in batch])
        processed_seg = torch.utils.data.dataloader.default_collate([item[2] for item in batch])
        processed_normal = torch.utils.data.dataloader.default_collate([item[3] for item in batch])
        #processed_pd_data = [(torch.tensor(item[4][0], dtype = torch.float32), torch.tensor(item[4][0], dtype = torch.float32)) for item in batch]
        processed_pd_data = [(item[4][0], item[4][1]) for item in batch]
        return processed_pointcloud, processed_cls, processed_seg, processed_normal, processed_pd_data


if __name__ == '__main__':
    train = PartNormalDataset(npoints=2048, split='trainval', normalize=False)
    test = PartNormalDataset(npoints=2048, split='test', normalize=False)
    #for data, label, _, _, pd_data in train:
    #    print(data.shape)
    #    print(label.shape)
    print(train[0][5])