import os
from collections import Counter

import h5py
import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Memory
from joblib import Parallel
from scipy.signal import resample_poly
from tqdm import tqdm

from deepsleep.sleep_stage import print_n_samples_each_class
from deepsleep.utils import get_balance_class_oversample

import re

cache_dir = './.cache'
memory = Memory(cache_dir, verbose=0, mmap_mode='r')

@memory.cache
def load_h5_file(filename):

    data = h5py.File(filename, 'r')
    x = data['data'][0, :, :]
    x = resample_poly(x, axis=1, up=100, down=128)[:, :, np.newaxis, np.newaxis].astype(np.float32)
    y = data['hypnogram'][:].astype(np.int32)
    return x, y, 100


all_bar_funcs = {
    'tqdm': lambda args: lambda x: tqdm(x, **args),
    # 'txt': lambda args: lambda x: text_progessbar(x, **args),
    'False': lambda args: iter,
    'None': lambda args: iter,
}


def ParallelExecutor(use_bar='tqdm', **joblib_args):
    def aprun(bar=use_bar, **tq_args):
        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                raise ValueError("Value %s not supported as bar type" % bar)
            return Parallel(**joblib_args)(bar_func(op_iter))
        return tmp
    return aprun


class NonSeqDataLoader(object):

    def __init__(self, data_dir, n_folds, fold_idx):
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.df = self._load_csv_files()

    def _load_csv_files(self):
        csv_files = os.listdir("data/csv")
        df = pd.concat([pd.read_csv(os.path.join("data", "csv", csv), index_col=0) for csv in csv_files])
        df = df.loc[pd.isnull(df.Skip)].reset_index(drop=True)
        h5files = os.listdir("data/h5")
        df = df.loc[df.FileID.str.lower().isin([f[:-3] for f in h5files]), :] \
               .sort_values(by=['Cohort', 'FileID']) \
               .reset_index(drop=True)
        n_test, n_train, n_eval = Counter([str(v) for v in df.Partition.values]).values()
        print "Number of train subjects: {}".format(n_train)
        print "Number of eval subjects: {}".format(n_eval)

        return df


    def _load_npz_file(self, npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def _load_npz_list_files(self, npz_files):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            print "Loading {} ...".format(npz_f)
            tmp_data, tmp_labels, sampling_rate = self._load_npz_file(npz_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")
            data.append(tmp_data)
            labels.append(tmp_labels)
        data = np.vstack(data)
        labels = np.hstack(labels)
        return data, labels

    def _load_h5_list_files(self, h5_files):
        """Load data and labels from list of h5 files."""

        data = ParallelExecutor(n_jobs=1)(total=len(h5_files))(
            delayed(load_h5_file)(filename=filename) for filename in h5_files
        )
        index_to_record = [idx for idx, v in enumerate(data) for _ in range(v[0].shape[0])]
        index_to_record_idx = [idx for _, v in enumerate(data) for idx in range(v[0].shape[0])]
        x = [d[0] for d in data]
        y = [d[1] for d in data]

        return x, y, {'record': index_to_record, 'idx': index_to_record_idx}

    def _load_cv_data(self, list_files):
        """Load training and cross-validation sets."""
        # Split files for training and validation sets
        val_files = np.array_split(list_files, self.n_folds)
        train_files = np.setdiff1d(list_files, val_files[self.fold_idx])

        # Load a npz file
        print "Load training set:"
        data_train, label_train = self._load_npz_list_files(train_files)
        print " "
        print "Load validation set:"
        data_val, label_val = self._load_npz_list_files(val_files[self.fold_idx])
        print " "

        # Reshape the data to match the input of the model - conv2d
        data_train = np.squeeze(data_train)
        data_val = np.squeeze(data_val)
        data_train = data_train[:, :, np.newaxis, np.newaxis]
        data_val = data_val[:, :, np.newaxis, np.newaxis]

        # Casting
        data_train = data_train.astype(np.float32)
        label_train = label_train.astype(np.int32)
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)

        return data_train, label_train, data_val, label_val

    def load_train_data(self, n_files=None):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        if n_files is not None:
            npzfiles = npzfiles[:n_files]

        subject_files = []
        for idx, f in enumerate(allfiles):
            if self.fold_idx < 10:
                pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
            if pattern.match(f):
                subject_files.append(os.path.join(self.data_dir, f))

        if len(subject_files) == 0:
            for idx, f in enumerate(allfiles):
                if self.fold_idx < 10:
                    pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]J0\.npz$".format(self.fold_idx))
                else:
                    pattern = re.compile("[a-zA-Z0-9]*{}[1-9]J0\.npz$".format(self.fold_idx))
                if pattern.match(f):
                    subject_files.append(os.path.join(self.data_dir, f))

        train_files = list(set(npzfiles) - set(subject_files))
        train_files.sort()
        subject_files.sort()

        # Load training and validation sets
        print "\n========== [Fold-{}] ==========\n".format(self.fold_idx)
        print "Load training set:"
        data_train, label_train = self._load_npz_list_files(npz_files=train_files)
        print " "
        print "Load validation set:"
        data_val, label_val = self._load_npz_list_files(npz_files=subject_files)
        print " "

        # Reshape the data to match the input of the model - conv2d
        data_train = np.squeeze(data_train)
        data_val = np.squeeze(data_val)
        data_train = data_train[:, :, np.newaxis, np.newaxis]
        data_val = data_val[:, :, np.newaxis, np.newaxis]

        # Casting
        data_train = data_train.astype(np.float32)
        label_train = label_train.astype(np.int32)
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)

        print "Training set: {}, {}".format(data_train.shape, label_train.shape)
        print_n_samples_each_class(label_train)
        print " "
        print "Validation set: {}, {}".format(data_val.shape, label_val.shape)
        print_n_samples_each_class(label_val)
        print " "

        # Use balanced-class, oversample training set
        x_train, y_train = get_balance_class_oversample(
            x=data_train, y=label_train
        )
        print "Oversampled training set: {}, {}".format(
            x_train.shape, y_train.shape
        )
        print_n_samples_each_class(y_train)
        print " "

        return x_train, y_train, data_val, label_val

    def load_h5_train_data(self, n_files=None):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        h5files = []
        for idx, f in enumerate(allfiles):
            if ".h5" in f:
                h5files.append(os.path.join(self.data_dir, f))
        h5files.sort()

        if n_files is not None:
            h5files = h5files[:n_files]

        # df = self._load_csv_files()
        train_files = [f for f in h5files if self.df.loc[self.df['FileID'].str.lower() == f.split('/')[-1].split('.')[0], 'Partition'].values[0] == 'train']
        subject_files = [f for f in h5files if self.df.loc[self.df['FileID'].str.lower() == f.split('/')[-1].split('.')[0], 'Partition'].values[0] == 'eval']
        # subject_files = h5files[len(h5files)/self.n_folds * self.fold_idx : len(h5files)/self.n_folds * (self.fold_idx + 1)]

        # subject_files = []
        # for idx, f in enumerate(allfiles):
        #     if self.fold_idx < 10:
        #         pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
        #     else:
        #         pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
        #     if pattern.match(f):
        #         subject_files.append(os.path.join(self.data_dir, f))

        # if len(subject_files) == 0:
        #     for idx, f in enumerate(allfiles):
        #         if self.fold_idx < 10:
        #             pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]J0\.npz$".format(self.fold_idx))
        #         else:
        #             pattern = re.compile("[a-zA-Z0-9]*{}[1-9]J0\.npz$".format(self.fold_idx))
        #         if pattern.match(f):
        #             subject_files.append(os.path.join(self.data_dir, f))

        # train_files = list(set(h5files) - set(subject_files))
        train_files.sort()
        subject_files.sort()

        # Load training and validation sets
        print "\n========== [Fold-{}] ==========\n".format(self.fold_idx)
        print "Load training set:"
        data_train, label_train, index_to_record_train = self._load_h5_list_files(h5_files=train_files)
        print " "
        print "Load validation set:"
        data_val, label_val, index_to_record_val = self._load_h5_list_files(h5_files=subject_files)
        print " "

        # # Reshape the data to match the input of the model - conv2d
        # data_train = np.squeeze(data_train)
        # data_val = np.squeeze(data_val)
        # data_train = data_train[:, :, np.newaxis, np.newaxis]
        # data_val = data_val[:, :, np.newaxis, np.newaxis]

        # # Casting
        # data_train = data_train.astype(np.float32)
        # label_train = label_train.astype(np.int32)
        # data_val = data_val.astype(np.float32)
        # label_val = label_val.astype(np.int32)

        # data_train = [np.array(d) for d in data_train]
        # label_train = [np.array(l) for l in label_train]
        # data_val = [np.array(d) for d in data_val]
        # label_val = [np.array(l) for l in label_val]

        print "Training set: {}, {}".format((len(index_to_record_train['idx']), ) + data_train[0].shape, (len(index_to_record_train['idx']), ) + label_train[0].shape)
        print_n_samples_each_class(label_train)
        print " "
        print "Validation set: {}, {}".format((len(index_to_record_val['idx']), ) + data_val[0].shape, (len(index_to_record_val['idx']), ) + label_val[0].shape)
        print_n_samples_each_class(label_val)
        print " "

        # # Use balanced-class, oversample training set
        # x_train, y_train = get_balance_class_oversample(
        #     x=data_train, y=label_train
        # )
        # print "Oversampled training set: {}, {}".format(
        #     x_train.shape, y_train.shape
        # )
        # print_n_samples_each_class(y_train)
        # print " "

        return data_train, label_train, data_val, label_val, index_to_record_train, index_to_record_val

    def load_test_data(self):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        subject_files = []
        for idx, f in enumerate(allfiles):
            if self.fold_idx < 10:
                pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
            if pattern.match(f):
                subject_files.append(os.path.join(self.data_dir, f))
        subject_files.sort()

        print "\n========== [Fold-{}] ==========\n".format(self.fold_idx)

        print "Load validation set:"
        data_val, label_val = self._load_npz_list_files(subject_files)

        # Reshape the data to match the input of the model
        data_val = np.squeeze(data_val)
        data_val = data_val[:, :, np.newaxis, np.newaxis]

        # Casting
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)

        return data_val, label_val


class SeqDataLoader(object):

    def __init__(self, data_dir, n_folds, fold_idx):
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx

    def _load_csv_files(self):
        csv_files = os.listdir("data/csv")
        df = pd.concat([pd.read_csv(os.path.join("data", "csv", csv), index_col=0) for csv in csv_files])
        df = df.loc[pd.isnull(df.Skip)].reset_index(drop=True)
        h5files = os.listdir("data/h5")
        df = df.loc[df.FileID.str.lower().isin([f[:-3] for f in h5files]), :] \
               .sort_values(by=['Cohort', 'FileID']) \
               .reset_index(drop=True)
        n_test, n_train, n_eval = Counter([str(v) for v in df.Partition.values]).values()
        print "Number of train subjects: {}".format(n_train)
        print "Number of eval subjects: {}".format(n_eval)

        return df

    def _load_npz_file(self, npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def _load_npz_list_files(self, npz_files):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            print "Loading {} ...".format(npz_f)
            tmp_data, tmp_labels, sampling_rate = self._load_npz_file(npz_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            tmp_data = np.squeeze(tmp_data)
            tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]

            # # Reshape the data to match the input of the model - conv1d
            # tmp_data = tmp_data[:, :, np.newaxis]

            # Casting
            tmp_data = tmp_data.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)

            data.append(tmp_data)
            labels.append(tmp_labels)

        return data, labels

    def _load_h5_list_files(self, h5_files):
        """Load data and labels from list of h5 files."""

        data = ParallelExecutor(n_jobs=1)(total=len(h5_files))(
            delayed(load_h5_file)(filename=filename) for filename in h5_files
        )
        index_to_record = [idx for idx, v in enumerate(data) for _ in range(v[0].shape[0])]
        index_to_record_idx = [idx for _, v in enumerate(data) for idx in range(v[0].shape[0])]

        labels = [d[1] for d in data]
        data = [d[0] for d in data]

        return data, labels, {'record': index_to_record, 'idx': index_to_record_idx}

    def _load_cv_data(self, list_files):
        """Load sequence training and cross-validation sets."""
        # Split files for training and validation sets
        val_files = np.array_split(list_files, self.n_folds)
        train_files = np.setdiff1d(list_files, val_files[self.fold_idx])

        # Load a npz file
        print "Load training set:"
        data_train, label_train = self._load_npz_list_files(train_files)
        print " "
        print "Load validation set:"
        data_val, label_val = self._load_npz_list_files(val_files[self.fold_idx])
        print " "

        return data_train, label_train, data_val, label_val

    def load_test_data(self):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        # Files for validation sets
        val_files = np.array_split(npzfiles, self.n_folds)
        val_files = val_files[self.fold_idx]

        print "\n========== [Fold-{}] ==========\n".format(self.fold_idx)

        print "Load validation set:"
        data_val, label_val = self._load_npz_list_files(val_files)

        return data_val, label_val

    def load_h5_train_data(self, n_files=None):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        h5files = []
        for idx, f in enumerate(allfiles):
            if ".h5" in f:
                h5files.append(os.path.join(self.data_dir, f))
        h5files.sort()

        if n_files is not None:
            h5files = h5files[:n_files]

        subject_files = h5files[len(h5files)/self.n_folds * self.fold_idx : len(h5files)/self.n_folds * (self.fold_idx + 1)]

        # subject_files = []
        # for idx, f in enumerate(allfiles):
        #     if self.fold_idx < 10:
        #         pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
        #     else:
        #         pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
        #     if pattern.match(f):
        #         subject_files.append(os.path.join(self.data_dir, f))

        train_files = list(set(h5files) - set(subject_files))
        train_files.sort()
        subject_files.sort()

        # Load training and validation sets
        print "\n========== [Fold-{}] ==========\n".format(self.fold_idx)
        print "Load training set:"
        data_train, label_train, index_to_record_train = self._load_h5_list_files(train_files)
        print " "
        print "Load validation set:"
        data_val, label_val, index_to_record_val = self._load_h5_list_files(subject_files)
        print " "

        print "Training set: n_subjects={}".format(len(data_train))
        n_train_examples = 0
        for d in data_train:
            print d.shape
            n_train_examples += d.shape[0]
        print "Number of examples = {}".format(n_train_examples)
        print_n_samples_each_class(label_train)
        print " "
        print "Validation set: n_subjects={}".format(len(data_val))
        n_valid_examples = 0
        for d in data_val:
            print d.shape
            n_valid_examples += d.shape[0]
        print "Number of examples = {}".format(n_valid_examples)
        print_n_samples_each_class(label_val)
        print " "

        return data_train, label_train, data_val, label_val

    def load_train_data(self, n_files=None):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        if n_files is not None:
            npzfiles = npzfiles[:n_files]

        subject_files = []
        for idx, f in enumerate(allfiles):
            if self.fold_idx < 10:
                pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
            if pattern.match(f):
                subject_files.append(os.path.join(self.data_dir, f))

        train_files = list(set(npzfiles) - set(subject_files))
        train_files.sort()
        subject_files.sort()

        # Load training and validation sets
        print "\n========== [Fold-{}] ==========\n".format(self.fold_idx)
        print "Load training set:"
        data_train, label_train = self._load_npz_list_files(train_files)
        print " "
        print "Load validation set:"
        data_val, label_val = self._load_npz_list_files(subject_files)
        print " "

        print "Training set: n_subjects={}".format(len(data_train))
        n_train_examples = 0
        for d in data_train:
            print d.shape
            n_train_examples += d.shape[0]
        print "Number of examples = {}".format(n_train_examples)
        print_n_samples_each_class(np.hstack(label_train))
        print " "
        print "Validation set: n_subjects={}".format(len(data_val))
        n_valid_examples = 0
        for d in data_val:
            print d.shape
            n_valid_examples += d.shape[0]
        print "Number of examples = {}".format(n_valid_examples)
        print_n_samples_each_class(np.hstack(label_val))
        print " "

        return data_train, label_train, data_val, label_val

    @staticmethod
    def load_h5_subject_data(data_dir, subject_idx):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(data_dir)
        h5files = []
        for idx, f in enumerate(allfiles):
            if ".h5" in f:
                h5files.append(os.path.join(data_dir, f))
        h5files.sort()

        subject_files = [h5files[subject_idx]]

        # Files for validation sets
        if len(subject_files) == 0 or len(subject_files) > 2:
            raise Exception("Invalid file pattern")

        def load_h5_list_files(h5_files):
            """Load data and labels from list of h5 files."""

            data = ParallelExecutor(n_jobs=1)(total=len(h5_files))(
                delayed(load_h5_file)(filename=filename) for filename in h5_files
            )
            index_to_record = [idx for idx, v in enumerate(data) for _ in range(v[0].shape[0])]
            index_to_record_idx = [idx for _, v in enumerate(data) for idx in range(v[0].shape[0])]

            labels = [d[1] for d in data]
            data = [d[0] for d in data]

            return np.array(data), np.array(labels), {'record': index_to_record, 'idx': index_to_record_idx}

        def load_npz_file(npz_file):
            """Load data and labels from a npz file."""
            with np.load(npz_file) as f:
                data = f["x"]
                labels = f["y"]
                sampling_rate = f["fs"]
            return data, labels, sampling_rate

        def load_npz_list_files(npz_files):
            """Load data and labels from list of npz files."""
            data = []
            labels = []
            fs = None
            for npz_f in npz_files:
                print "Loading {} ...".format(npz_f)
                tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
                if fs is None:
                    fs = sampling_rate
                elif fs != sampling_rate:
                    raise Exception("Found mismatch in sampling rate.")

                # Reshape the data to match the input of the model - conv2d
                tmp_data = np.squeeze(tmp_data)
                tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]

                # # Reshape the data to match the input of the model - conv1d
                # tmp_data = tmp_data[:, :, np.newaxis]

                # Casting
                tmp_data = tmp_data.astype(np.float32)
                tmp_labels = tmp_labels.astype(np.int32)

                data.append(tmp_data)
                labels.append(tmp_labels)

            return data, labels

        print "Load data from: {}".format(subject_files)
        data, labels, _ = load_h5_list_files(subject_files)

        return data, labels

    @staticmethod
    def load_subject_data(data_dir, subject_idx):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(data_dir)
        subject_files = []
        for idx, f in enumerate(allfiles):
            if subject_idx < 10:
                pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(subject_idx))
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(subject_idx))
            if pattern.match(f):
                subject_files.append(os.path.join(data_dir, f))

        # Files for validation sets
        if len(subject_files) == 0 or len(subject_files) > 2:
            raise Exception("Invalid file pattern")

        def load_npz_file(npz_file):
            """Load data and labels from a npz file."""
            with np.load(npz_file) as f:
                data = f["x"]
                labels = f["y"]
                sampling_rate = f["fs"]
            return data, labels, sampling_rate

        def load_npz_list_files(npz_files):
            """Load data and labels from list of npz files."""
            data = []
            labels = []
            fs = None
            for npz_f in npz_files:
                print "Loading {} ...".format(npz_f)
                tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
                if fs is None:
                    fs = sampling_rate
                elif fs != sampling_rate:
                    raise Exception("Found mismatch in sampling rate.")

                # Reshape the data to match the input of the model - conv2d
                tmp_data = np.squeeze(tmp_data)
                tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]

                # # Reshape the data to match the input of the model - conv1d
                # tmp_data = tmp_data[:, :, np.newaxis]

                # Casting
                tmp_data = tmp_data.astype(np.float32)
                tmp_labels = tmp_labels.astype(np.int32)

                data.append(tmp_data)
                labels.append(tmp_labels)

            return data, labels

        print "Load data from: {}".format(subject_files)
        data, labels = load_npz_list_files(subject_files)

        return data, labels
