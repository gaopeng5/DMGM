#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
import random
import numpy as np
from scipy.io import loadmat
import scipy.spatial
from PIL import Image

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
import torchvision.models as models
import torchvision.transforms as T

loc_noise = 1.2    #5   #1.2
yaw_noise = 0.2    #1   #0.2


class FaceToEdge(object):
    r"""Converts mesh faces :obj:`[3, num_faces]` to edge indices
    :obj:`[2, num_edges]`.

    Args:
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed.
    """

    def __init__(self, remove_faces=True):
        self.remove_faces = remove_faces

    def __call__(self, data):
        if data.face is not None:
            face = data.face
            edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

            data.edge_index = edge_index
            if self.remove_faces:
                data.face = None

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class Distance(object):
    r"""Saves the Euclidean distance of linked nodes in its edge attributes.

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized to the interval :math:`[0, 1]`. (default: :obj:`True`)
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of the maximum value
            found in the data. (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
    """
    def __init__(self, norm=True, max_value=None, cat=True):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)

        if self.norm and dist.numel() > 0:
            dist = dist / (dist.max() if self.max is None else self.max)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = dist

        return data

    def __repr__(self):
        return '{}(norm={}, max_value={})'.format(self.__class__.__name__,
                                                  self.norm, self.max)


class Delaunay(object):
    r"""Computes the delaunay triangulation of a set of points."""
    def __call__(self, data):
        if data.pos.size(0) < 2:
            data.edge_index = torch.tensor([], dtype=torch.long,
                                           device=data.pos.device).view(2, 0)
        if data.pos.size(0) == 2:
            data.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long,
                                           device=data.pos.device)
        elif data.pos.size(0) == 3:
            data.face = torch.tensor([[0], [1], [2]], dtype=torch.long,
                                     device=data.pos.device)
        if data.pos.size(0) > 3:
            pos = data.pos.cpu().numpy()
            tri = scipy.spatial.Delaunay(pos[:,:2], qhull_options='QJ')
            face = torch.from_numpy(tri.simplices)

            data.face = face.t().contiguous().to(data.pos.device, torch.long)

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def x_to_world_transformation(gps):
    # gps is a list in the format of [x, y, z, pitch, roll, yaw]

    # used for rotation matrix
    c_y = np.cos(np.radians(gps[5]))
    s_y = np.sin(np.radians(gps[5]))
    c_r = np.cos(np.radians(gps[4]))
    s_r = np.sin(np.radians(gps[4]))
    c_p = np.cos(np.radians(gps[3]))
    s_p = np.sin(np.radians(gps[3]))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = gps[0]
    matrix[1, 3] = gps[1]
    matrix[2, 3] = gps[2]

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


# train data uses accurate gps, test data adds noise to gps
class traindata(InMemoryDataset):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # batch_size = 1

    default_transform = T.Compose([Delaunay(), FaceToEdge(), Distance()])

    def __init__(self, in_folder, out_folder, category, 
                 transform=default_transform, pre_transform=None, pre_filter=None):
        #assert category.lower() in self.categories
        self.category = category
        self.in_folder = in_folder
        self.out_folder = out_folder
        super().__init__(out_folder, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.in_folder)
    @property
    def processed_dir(self):
        return os.path.join(self.out_folder, self.category.capitalize())
    @property
    def raw_file_names(self):
        return [self.category]

    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def process(self):
        if models is None or T is None or Image is None:
            raise ImportError('Package `torchvision` could not be found.')

        category = self.category.capitalize()
        names = glob.glob(os.path.join(self.raw_dir, category, '*.png'))
        name = names[0][:-4]       

        matdata = loadmat('{}.mat'.format(name))
        pos = torch.from_numpy(matdata['pos3D']).to(torch.float)
        pos2d = torch.from_numpy(matdata['pos'][:,0:2]).to(torch.float)         
        gt = torch.from_numpy(matdata['y']).to(torch.float).view(-1)
        feats = torch.from_numpy(matdata['X']).to(torch.float)
        
        gps = matdata['GPS'][0, 0:6]
        # assume x,y,yaw has a normalized noise
        gps[0] += np.random.normal(0, loc_noise)
        gps[1] += np.random.normal(0, loc_noise)
        gps[5] += np.random.normal(0, yaw_noise)
        # transformation matrix for conversion from vehicle view to world
        transformation = torch.from_numpy(
            x_to_world_transformation(gps)).to(torch.float)
        # convert from sensor coordinates to Unreal Engine coordinates
        # (x, y, z) -> (z, x, -y)
        p3d = torch.stack([pos[:,2], pos[:,0], -pos[:,1], 
                           torch.ones(pos.shape[0])], dim=1)
        pos_world = torch.matmul(transformation, torch.transpose(p3d, 0, 1))
        pos_world = torch.transpose(pos_world, 0, 1)[:,:3]
 
        label = torch.from_numpy(matdata['class']).to(torch.int).view(-1)

        data_list = [Data(pos=pos, name=name, y=gt, x=feats, 
                          norm=pos2d, pos_world=pos_world, label=label)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data,slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({}, category={})'.format(self.__class__.__name__, len(self),
                                            self.category)


# train data uses accurate gps, test data adds noise to gps
class testdata(InMemoryDataset):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # batch_size = 1

    default_transform = T.Compose([Delaunay(), FaceToEdge(), Distance()])

    def __init__(self, in_folder, out_folder, category, 
                 transform=default_transform, pre_transform=None, pre_filter=None):
        #assert category.lower() in self.categories
        self.category = category
        self.in_folder = in_folder
        self.out_folder = out_folder
        super().__init__(out_folder, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.in_folder)
    @property
    def processed_dir(self):
        return os.path.join(self.out_folder, self.category.capitalize())
    @property
    def raw_file_names(self):
        return [self.category]

    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def process(self):
        if models is None or T is None or Image is None:
            raise ImportError('Package `torchvision` could not be found.')

        category = self.category.capitalize()
        names = glob.glob(os.path.join(self.raw_dir, category, '*.png'))
        name = names[0][:-4]       

        matdata = loadmat('{}.mat'.format(name))
        pos = torch.from_numpy(matdata['pos3D']).to(torch.float)
        pos2d = torch.from_numpy(matdata['pos'][:,0:2]).to(torch.float)         
        gt = torch.from_numpy(matdata['y']).to(torch.float).view(-1)
        feats = torch.from_numpy(matdata['X']).to(torch.float)
        
        gps = matdata['GPS'][0, 0:6]
        # assume x,y,yaw has a normalized noise
        gps[0] += np.random.normal(0, loc_noise)
        gps[1] += np.random.normal(0, loc_noise)
        gps[5] += np.random.normal(0, yaw_noise)
        # transformation matrix for conversion from vehicle view to world
        transformation = torch.from_numpy(
            x_to_world_transformation(gps)).to(torch.float)
        # convert from sensor coordinates to Unreal Engine coordinates
        # (x, y, z) -> (z, x, -y)
        p3d = torch.stack([pos[:,2], pos[:,0], -pos[:,1], 
                           torch.ones(pos.shape[0])], dim=1)
        pos_world = torch.matmul(transformation, torch.transpose(p3d, 0, 1))
        pos_world = torch.transpose(pos_world, 0, 1)[:,:3]
 
        label = torch.from_numpy(matdata['class']).to(torch.int).view(-1)

        data_list = [Data(pos=pos, name=name, y=gt, x=feats, 
                          norm=pos2d, pos_world=pos_world, label=label)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data,slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({}, category={})'.format(self.__class__.__name__, len(self),
                                            self.category)


class PairData(Data):  # pragma: no cover
    def __inc__(self, key, value, *args):
        if bool(re.search('index_s', key)):
            return self.x_s.size(0)
        if bool(re.search('index_t', key)):
            return self.x_t.size(0)
        else:
            return 0


def switch_data(data_s, data_t):
    if data_s['y'].size() > data_t['y'].size():
        return True, data_t, data_s

    return False, data_s, data_t


def obtain_y(t1, t2):
    """
    gt is a list with the same length of t1
    each element in gt indicates the index of co-visible objects in t2
    for example, t1 is a list of 6 elements, 
    if gt = [5, 100, 100, 100, 100, 100], this means only 1 object is 
    co-visible in t1 and t2, and this object is the first element in t1, 
    and the 6th element in t2
    """
    gt = torch.zeros(t1.size(), dtype=torch.long)
    for i, data in enumerate(t1):
        tmp = (t2 == data).nonzero()
        if tmp.size(0) == 0:
            gt[i] = 100
        else:
            gt[i] = tmp
    return gt


class PairDataset(torch.utils.data.Dataset):
    r"""Combines two datasets, a source dataset and a target dataset, by
    building pairs between separate dataset examples.

    Args:
        dataset_s (torch.utils.data.Dataset): The source dataset.
        dataset_t (torch.utils.data.Dataset): The target dataset.
        sample (bool, optional): If set to :obj:`True`, will sample exactly
            one target example for every source example instead of holding the
            product of all source and target examples. (default: :obj:`False`)
    """
    def __init__(self, dataset_s, dataset_t, sample=False):
        self.dataset_s = dataset_s
        self.dataset_t = dataset_t
        self.sample = sample

    def __len__(self):
        return len(self.dataset_s) if self.sample else len(
            self.dataset_s) * len(self.dataset_t)
    
    def __getitem__(self, idx):
        if self.sample:
            data_s = self.dataset_s[idx]
            data_t = self.dataset_t[random.randint(0, len(self.dataset_t) - 1)]
        else:
            data_s = self.dataset_s[idx]
            data_t = self.dataset_t[idx]
        
        switched, data_s, data_t = switch_data(data_s, data_t)
        gt = obtain_y(data_s.y, data_t.y)
        
        return PairData(
            x_s=data_s.x,
            edge_index_s=data_s.edge_index,
            edge_attr_s=data_s.edge_attr,
            pos_s=data_s.pos,
            name_s=data_s.name,
            pos_world_s=data_s.pos_world,
            x_t=data_t.x,
            edge_index_t=data_t.edge_index,
            edge_attr_t=data_t.edge_attr,
            pos_t=data_t.pos,
            name_t=data_t.name,
            pos_world_t=data_t.pos_world,
            num_nodes=None,
            y=gt,
            switched=switched
        )

    def __repr__(self):
        return '{}({}, {}, sample={})'.format(self.__class__.__name__,
                                              self.dataset_s, self.dataset_t,
                                              self.sample)



