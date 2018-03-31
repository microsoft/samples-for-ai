#coding=utf-8

import os
import numpy as np
import h5py
import threading
import queue

#----------------------------------------------------------------------------

class Dataset:
    def __init__(self,
        h5_path,                                # e.g. 'cifar10-32.h5'
        resolution                  = None,     # e.g. 32 (autodetect if None)
        label_path                  = None,     # e.g. 'cifar100-32-labels.npy' (autodetect if None)
        mirror_augment              = False,
        max_images                  = None,
        max_labels                  = 'all',
        shuffle                     = True,
        prefetch_images             = 512,
        max_gb_to_load_right_away   = 4):

        # Open HDF5 file and select resolution.
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, 'r')
        self.resolution = resolution
        if self.resolution is None:
            self.resolution = max(value.shape[1] for key, value in self.h5_file.items() if key.startswith('data'))

        # Initialize LODs.
        self.resolution_log2 = int(np.floor(np.log2(self.resolution)))
        assert self.resolution == 2 ** self.resolution_log2
        self.lod_resolutions = [2 ** i for i in range(self.resolution_log2, -1, -1)]
        self.h5_lods = [self.h5_file['data%dx%d' % (r, r)] for r in self.lod_resolutions]

        # Look up shapes and dtypes.
        
        self.shape = self.h5_lods[0].shape



        if max_images is not None:
            self.shape = (min(self.shape[0], max_images),) + self.shape[1:]

        #change for channel last
        self.dtype = self.h5_lods[0].dtype
        self.lod_shapes = [(self.shape[0],  r, r, self.shape[3]) for r in self.lod_resolutions]
        
        assert min(self.shape) > 0
        assert all(lod.shape[1:] == shape[1:] for lod, shape in zip(self.h5_lods, self.lod_shapes))
        assert all(lod.dtype == self.dtype for lod in self.h5_lods)

        # Initialize shuffling and prefetching.
        self.mirror_augment = mirror_augment
        self.prefetch_images = max(prefetch_images, 2)
        self.max_gb_to_load_right_away = max_gb_to_load_right_away
        min_order_size = self.prefetch_images * 4
        order_size = self.shape[0] * ((min_order_size - 1) / self.shape[0] + 1)
        self.order = (np.arange(order_size) % self.shape[0]).astype(int)
        if shuffle:
            np.random.shuffle(self.order)
            self.reshuffle_window = min(self.order.size / 2, self.order.size - self.prefetch_images * 2 - 1)
        else:
            self.reshuffle_window = 1
        self.queue = queue.Queue(self.prefetch_images)
        self.thread = None
        self.cur_pos = 0
        self.cur_lod = -1

        # Autodetect label path.
        self.label_path = label_path
        if self.label_path is None:
            tmp = os.path.splitext(self.h5_path)[0] + '-labels.npy'
            if os.path.isfile(tmp):
                self.label_path = tmp

        # Load labels.
        if self.label_path is None or max_labels == 0:
            self.labels = np.zeros((self.shape[0], 0), dtype=np.float32)
        else:
            assert self.label_path.endswith('.npy')
            self.labels = np.load(self.label_path)
        if max_labels is not None and max_labels != 'all':
            if self.labels.shape[1] > max_labels:
                self.labels = self.labels[:, :max_labels]

    def close(self):
        self.kill_worker_thread()
        self.h5_file.close()

    def get_dynamic_range(self): # [min, max]
        assert self.dtype == np.uint8
        return [0, 255]

    def get_images(self):
        return self.h5_lods[0][:self.shape[0]]

    def get_random_minibatch(self, minibatch_size, lod=0, shrink_based_on_lod=False, labels=False):
        assert minibatch_size >= 1
        lod = np.clip(float(lod), 0.0, float(self.resolution_log2))
        lod_int = int(np.floor(lod))

        # LOD changed => kill previous worker thread.
        if lod_int != self.cur_lod:
            self.kill_worker_thread()
            self.cur_lod = lod_int

        # No worker thread => launch one.
        if self.thread is None:
            while not self.queue.empty():
                self.queue.get()
            h5_lod = self.h5_lods[lod_int]
            total_gb = np.prod(np.float64(self.lod_shapes[lod_int])) * np.dtype(self.dtype).itemsize / np.exp2(30)
            if total_gb <= self.max_gb_to_load_right_away:
                h5_lod = h5_lod[:self.shape[0]] # load all data right away
            self.thread = WorkerThread(h5_lod, self.queue, self.order, self.cur_pos)
            self.thread.daemon = True
            self.thread.start()

        # Grab data from worker thread.
        data = np.stack([self.queue.get() for i in range(minibatch_size)])

        # Reshuffle indices.
        ivec = (np.arange(minibatch_size) + self.cur_pos) % self.order.size
        jvec = (ivec - np.random.randint(self.reshuffle_window, size=minibatch_size)) % self.order.size
        orig_indices = self.order[ivec]
        for i, j in zip(ivec, jvec):
            self.order[i], self.order[j] = self.order[j], self.order[i]
        self.cur_pos = (self.cur_pos + minibatch_size) % self.order.size

        # Apply mirror augment.
        if self.mirror_augment:
            mask = np.random.rand(data.shape[0]) < 0.5
            data[mask] = data[mask,  :, ::-1, :]

       
        
        # Apply fractional LOD.
        if lod != lod_int:
            n, c, h, w = data.shape
            t = data.reshape(n,  h/2, 2, w/2, 2, c).mean((3, 5)).repeat(2, 2).repeat(2, 3)
            
            data = (data + (t - data) * (lod - lod_int)).astype(self.dtype)
        if not shrink_based_on_lod and lod_int != 0:
            data = data.repeat(2 ** lod_int, 2).repeat(2 ** lod_int, 3)

        data=np.swapaxes(data,1,2)
        data=np.swapaxes(data,2,3)
        
        if labels:
            return data, self.labels[orig_indices]
        else:
            return data

    def get_random_minibatch_channel_last(self, minibatch_size, lod=0, shrink_based_on_lod=False, labels=False):
        assert minibatch_size >= 1
        lod = np.clip(float(lod), 0.0, float(self.resolution_log2))
        lod_int = int(np.floor(lod))

        # LOD changed => kill previous worker thread.
        if lod_int != self.cur_lod:
            self.kill_worker_thread()
            self.cur_lod = lod_int

        # No worker thread => launch one.
        if self.thread is None:
            while not self.queue.empty():
                self.queue.get()
            h5_lod = self.h5_lods[lod_int]
            total_gb = np.prod(np.float64(self.lod_shapes[lod_int])) * np.dtype(self.dtype).itemsize / np.exp2(30)
            if total_gb <= self.max_gb_to_load_right_away:
                h5_lod = h5_lod[:self.shape[0]] # load all data right away
            self.thread = WorkerThread(h5_lod, self.queue, self.order, self.cur_pos)
            self.thread.daemon = True
            self.thread.start()

        # Grab data from worker thread.
        data = np.stack([self.queue.get() for i in range(minibatch_size)])

        # Reshuffle indices.
        ivec = (np.arange(minibatch_size) + self.cur_pos) % self.order.size
        jvec = (ivec - np.random.randint(self.reshuffle_window, size=minibatch_size)) % self.order.size
        orig_indices = self.order[ivec]
        for i, j in zip(ivec, jvec):
            self.order[i], self.order[j] = self.order[j], self.order[i]
        self.cur_pos = (self.cur_pos + minibatch_size) % self.order.size

        # Apply mirror augment.

        #change to channal last 
        if self.mirror_augment:
            mask = np.random.rand(data.shape[0]) < 0.5
            data[mask] = data[mask,  :, ::-1,:]


        # Apply fractional LOD.
        #change to channel last
        if lod != lod_int:
            n, h, w, c = data.shape
            t = data.reshape(n,  int(h/2), 2, int(w/2), 2, c ).mean((2, 4)).repeat(2, 1).repeat(2, 2)
            
            data = (data + (t - data) * (lod - lod_int)).astype(self.dtype)
        if not shrink_based_on_lod and lod_int != 0:
            data = data.repeat(2 ** lod_int, 1).repeat(2 ** lod_int, 2)

        # Look up labels.

        #print("****************************************")
        #print("minibach.shape：",data.shape)
       
        if labels:
            return data, self.labels[orig_indices]
        else:
            return data

    def kill_worker_thread(self):
        if self.thread is not None:
            self.thread.exit_requested = True
            while not self.queue.empty():
                self.queue.get()
            self.thread.join()
            self.thread = None

#----------------------------------------------------------------------------

class WorkerThread(threading.Thread):
    def __init__(self, dataset, queue, order, start_pos):
        threading.Thread.__init__(self)
        #print("此处使用了dataset")
        self.dataset = dataset
        self.queue = queue
        self.order = order
        self.cur_pos = start_pos
        self.exit_requested = False

    def run(self):
        while not self.exit_requested:
            #print(self.cur_pos)
            #print(self.order[self.cur_pos])
            data = self.dataset[int(self.order[int(self.cur_pos)])]
            self.queue.put(data)
            self.cur_pos = (self.cur_pos + 1) % self.order.size

#----------------------------------------------------------------------------
