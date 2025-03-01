# original data



from torch.utils.data import Dataset
from petrel_client.client import Client
from tqdm import tqdm
import numpy as np
import io
import time
import xarray as xr
import json
import pandas as pd
import os
import gc
from multiprocessing import Pool
from multiprocessing import shared_memory
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import copy
import queue
import torch
import warnings
# from s3_client import s3_client

Years = {
    'train': ['1979-01-01 00:00:00', '2015-12-31 23:00:00'],
    'valid': ['2018-01-01 00:00:00', '2018-12-31 23:00:00'],
    'test': ['2016-01-01 00:00:00', '2017-12-31 23:00:00'],
    'all': ['1979-01-01 00:00:00', '2020-12-31 23:00:00']
}

multi_level_vnames = [
    "z", "t", "q", "r", "u", "v", "vo", "pv",
]
single_level_vnames = [
    "t2m", "u10", "v10", "tcc", "tp", "tisr",
]
long2shortname_dict = {"geopotential": "z", "temperature": "t", "specific_humidity": "q", "relative_humidity": "r", "u_component_of_wind": "u", "v_component_of_wind": "v", "vorticity": "vo", "potential_vorticity": "pv", \
    "2m_temperature": "t2m", "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10", "total_cloud_cover": "tcc", "total_precipitation": "tp", "toa_incident_solar_radiation": "tisr"}
constants = [
    "lsm", "slt", "orography"
]
height_level = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, \
    500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
# height_level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

multi_level_dict_param = {"z":height_level, "t": height_level, "q": height_level, "r": height_level}
multi_acc_mean_dir = "era5_new:s3://era5_np_float32/climate_mean_day/1993-2016"
single_acc_mean_dir = "era5_new:s3://era5_np_float32/climate_mean_day/single/1993-2016"

def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma



class era5_npy_f32(Dataset):
    def __init__(self, data_dir='era5_new:s3://era5_np_float32', split='train', **kwargs) -> None:
        super().__init__()
        # print("init begin")
        self.length = kwargs.get('length', 1)
        self.file_stride = kwargs.get('file_stride', 6)
        self.sample_stride = kwargs.get('sample_stride', 1)
        self.output_meanstd = kwargs.get("output_meanstd", False)
        self.pred_length = kwargs.get("pred_length", None)
        self.use_diff_pos = kwargs.get("use_diff_pos", False)
        self.rm_equator = kwargs.get("rm_equator", False)
        Years_dict = kwargs.get('years', Years)
        process_num = kwargs.get("process_num", 60)


        vnames_type = kwargs.get("vnames", {})
        self.constants_types = vnames_type.get('constants', [])
        self.single_level_vnames = vnames_type.get('single_level_vnames', ['u10', 'v10', 't2m', 'msl'])
        self.multi_level_vnames = vnames_type.get('multi_level_vnames', ['z','q', 'u', 'v', 't'])
        self.height_level_list = vnames_type.get('hight_level_list', [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
        self.height_level_indexes = [height_level.index(j) for j in self.height_level_list]

        self.select_row = [i for i in range(721)]
        if self.rm_equator:
            del self.select_row[360]
        self.split = split
        self.data_dir = data_dir
        self.client = Client(conf_path="~/petreloss.conf")
        # self.client = Client(conf_path="~/petreloss.conf")
        years = Years_dict[split]
        self.init_file_list(years)

        # print(constants_index)
        if len(self.constants_types) > 0:
            self.constants_data = self.get_constants_data(self.constants_types)
        else:
            self.constants_data = None
        self._get_meanstd()
        self.mean, self.std = self.get_meanstd()

        self.data_element_num = len(self.single_level_vnames) + len(self.multi_level_vnames) * len(self.height_level_list)
        dim = len(self.single_level_vnames) + len(self.multi_level_vnames) * len(self.height_level_list)



        self.index_dict1 = {}
        self.index_dict2 = {}
        i = 0
        for vname in self.single_level_vnames:
            self.index_dict1[(vname, 0)] = i
            i += 1
        for vname in self.multi_level_vnames:
            for height in self.height_level_list:
                self.index_dict1[(vname, height)] = i
                i += 1

        self.index_queue = multiprocessing.Queue()
        self.unit_data_queue = multiprocessing.Queue()

        self.index_queue.cancel_join_thread() 
        self.unit_data_queue.cancel_join_thread()

        self.compound_data_queue = []
        self.sharedmemory_list = []
        self.compound_data_queue_dict = {}
        self.sharedmemory_dict = {}


        self.compound_data_queue_num = 8

        self.lock = multiprocessing.Lock()
        if self.rm_equator:
            self.a = np.zeros((dim, 720, 1440), dtype=np.float32)
        else:
            self.a = np.zeros((dim, 721, 1440), dtype=np.float32)


        for _ in range(self.compound_data_queue_num):
            self.compound_data_queue.append(multiprocessing.Queue())
            shm = shared_memory.SharedMemory(create=True, size=self.a.nbytes)
            shm.unlink()
            self.sharedmemory_list.append(shm)

        self.arr = multiprocessing.Array('i', range(self.compound_data_queue_num))



        self._workers = []

        for _ in range(process_num):
            w = multiprocessing.Process(
                target=self.load_data_process)
            w.daemon = True

            w.start()
            self._workers.append(w)
        w = multiprocessing.Process(target=self.data_compound_process)
        w.daemon = True
        w.start()
        self._workers.append(w)



    def init_file_list(self, years):
        time_sequence = pd.date_range(years[0],years[1],freq=str(self.file_stride)+'h') #pd.date_range(start='2019-1-09',periods=24,freq='H')
        self.file_list= [os.path.join(str(time_stamp.year), str(time_stamp.to_datetime64()).split('.')[0]).replace('T', '/')
                      for time_stamp in time_sequence]
        self.single_file_list= [os.path.join('single/'+str(time_stamp.year), str(time_stamp.to_datetime64()).split('.')[0]).replace('T', '/')
                      for time_stamp in time_sequence]


    def _get_meanstd(self):
        with open('./datasets/mean_std.json',mode='r') as f:
            multi_level_mean_std = json.load(f)
        with open('./datasets/mean_std_single.json',mode='r') as f:
            single_level_mean_std = json.load(f)
        self.mean_std = {}
        multi_level_mean_std['mean'].update(single_level_mean_std['mean'])
        multi_level_mean_std['std'].update(single_level_mean_std['std'])
        self.mean_std['mean'] = multi_level_mean_std['mean']
        self.mean_std['std'] = multi_level_mean_std['std']
        for vname in self.single_level_vnames:
            self.mean_std['mean'][vname] = np.array(self.mean_std['mean'][vname])[::-1][:,np.newaxis,np.newaxis]
            self.mean_std['std'][vname] = np.array(self.mean_std['std'][vname])[::-1][:,np.newaxis,np.newaxis]
        for vname in self.multi_level_vnames:
            self.mean_std['mean'][vname] = np.array(self.mean_std['mean'][vname])[::-1][:,np.newaxis,np.newaxis]
            self.mean_std['std'][vname] = np.array(self.mean_std['std'][vname])[::-1][:,np.newaxis,np.newaxis]
    

    def get_noise_weight(self):
        diff_pow2_mean_list = []
        for vname in self.single_level_vnames:

            url = f"{self.data_dir}/diff_mean_std/diff_pow2_mean_{vname}.npy"
            with io.BytesIO(self.client.get(url)) as f:
                unit_data = np.load(f)
            diff_pow2_mean_list.append(unit_data)
        
        for vname in self.multi_level_vnames:
            for height in self.height_level_list:

                url = f"{self.data_dir}/diff_mean_std/diff_pow2_mean_{vname}_{height}.npy"
                with io.BytesIO(self.client.get(url)) as f:
                    unit_data = np.load(f)
                diff_pow2_mean_list.append(unit_data)   
        
        diff_pow2_mean = np.stack(diff_pow2_mean_list, axis=0)
        del diff_pow2_mean_list
        return diff_pow2_mean.reshape(diff_pow2_mean.shape[0], -1).mean(axis=-1)[:,np.newaxis,np.newaxis]**0.5



    def get_diffmeanstd(self):
        diff_mean_list = []
        diff_pow2_mean_list = []
        for vname in self.single_level_vnames:
            url = f"{self.data_dir}/diff_mean_std/diff_mean_{vname}.npy"
            with io.BytesIO(self.client.get(url)) as f:
                unit_data = np.load(f)
            diff_mean_list.append(unit_data)

            
            url = f"{self.data_dir}/diff_mean_std/diff_pow2_mean_{vname}.npy"
            with io.BytesIO(self.client.get(url)) as f:
                unit_data = np.load(f)
            diff_pow2_mean_list.append(unit_data)
        
        for vname in self.multi_level_vnames:
            for height in self.height_level_list:
                url = f"{self.data_dir}/diff_mean_std/diff_mean_{vname}_{height}.npy"
                with io.BytesIO(self.client.get(url)) as f:
                    unit_data = np.load(f)
                diff_mean_list.append(unit_data)

                
                url = f"{self.data_dir}/diff_mean_std/diff_pow2_mean_{vname}_{height}.npy"
                with io.BytesIO(self.client.get(url)) as f:
                    unit_data = np.load(f)
                diff_pow2_mean_list.append(unit_data)   
        
        diff_mean = np.stack(diff_mean_list, axis=0)
        diff_pow2_mean = np.stack(diff_pow2_mean_list, axis=0)
        del diff_mean_list
        del diff_pow2_mean_list
        if self.use_diff_pos:
            diff_std = diff_pow2_mean - diff_mean ** 2
            return diff_mean, diff_std**0.5
        else:
            diff_std = diff_pow2_mean.reshape(diff_pow2_mean.shape[0], -1).mean(axis=-1) - diff_mean.reshape(diff_mean.shape[0], -1).mean(axis=-1)**2
            return diff_mean.reshape(diff_mean.shape[0], -1).mean(axis=-1)[:, np.newaxis, np.newaxis], diff_std[:,np.newaxis,np.newaxis]**0.5
          


    def get_constants_data(self, constants_types):
        file = os.path.join("constant", "z_lsm_slt.nc")
        url = f"era5:s3://era5_nc/{file}"
        array_lst = []
        with io.BytesIO(self.client.get(url)) as f:
            nc_data = xr.open_dataset(f)
            for vname in constants_types:
                D = nc_data.data_vars[vname].data
                D = standardization(D)
                array_lst.append(D[np.newaxis, :, :])
            data = np.concatenate(array_lst, axis=0)[:, self.select_row, :]
            array = data
        del array_lst
        return array


    def data_compound_process(self):
        recorder_dict = {}
        while True:
            job_pid, idx, vname, height = self.unit_data_queue.get()
            if job_pid not in self.compound_data_queue_dict:
                try:
                    self.lock.acquire()
                    for i in range(self.compound_data_queue_num):
                        if job_pid == self.arr[i]:
                            self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                            break
                    if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                        print("error", job_pid, self.arr)
                except Exception as err:
                    raise err
                finally:
                    self.lock.release()
            if (job_pid, idx) in recorder_dict:
                recorder_dict[(job_pid, idx)][(vname, height)] = 1
            else:
                recorder_dict[(job_pid, idx)] = {(vname, height): 1}
            if len(recorder_dict[(job_pid, idx)]) == self.data_element_num:
                del recorder_dict[(job_pid, idx)]
                self.compound_data_queue_dict[job_pid].put((idx))

    def put_index(self, idx):
        job_pid = os.getpid()
        if job_pid not in self.compound_data_queue_dict:
            try:
                self.lock.acquire()
                for i in range(self.compound_data_queue_num):
                    if i == self.arr[i]:
                        self.arr[i] = job_pid
                        self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                        self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                        break
                if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                    print("error", job_pid, self.arr)
  
            except Exception as err:
                raise err
            finally:
                self.lock.release()

        try:
            idx = self.compound_data_queue_dict[job_pid].get(False)
            raise ValueError
        except queue.Empty:
            pass
        except Exception as err:
            raise err
        
        for vname in self.single_level_vnames:
            self.index_queue.put((job_pid, idx, vname, 0))
        for vname in self.multi_level_vnames:
            for height in self.height_level_list:
                self.index_queue.put((job_pid, idx, vname, height))

        # self.index_queue.put((job_pid, idx, self.single_level_vnames))
        # self.index_queue.put((job_pid, idx, self.multi_level_vnames))


    def queue_wait_data(self):
        job_pid = os.getpid()
        return_data = {}
        idx = self.compound_data_queue_dict[job_pid].get()
        b = np.ndarray(self.a.shape, dtype=self.a.dtype, buffer=self.sharedmemory_dict[job_pid].buf)
        if self.constants_data is not None:
            return_data[idx] = np.concatenate((self.constants_data, b), axis=0)
        else:
            return_data[idx] = copy.deepcopy(b)
        return return_data


    def get_data(self, idxes):
        job_pid = os.getpid()
        if job_pid not in self.compound_data_queue_dict:
            try:
                self.lock.acquire()
                for i in range(self.compound_data_queue_num):
                    if i == self.arr[i]:
                        self.arr[i] = job_pid
                        self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                        self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                        break
                if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                    print("error", job_pid, self.arr)

                
            except Exception as err:
                raise err
            finally:
                self.lock.release()

        try:
            idx = self.compound_data_queue_dict[job_pid].get(False)
            raise ValueError
        except queue.Empty:
            pass
        except Exception as err:
            raise err
        
        b = np.ndarray(self.a.shape, dtype=self.a.dtype, buffer=self.sharedmemory_dict[job_pid].buf)
        return_data = {}
        for idx in idxes:
            for vname in self.single_level_vnames:
                self.index_queue.put((job_pid, idx, vname, 0))
            for vname in self.multi_level_vnames:
                for height in self.height_level_list:
                    self.index_queue.put((job_pid, idx, vname, height))
            idx = self.compound_data_queue_dict[job_pid].get()
            # b -= self.mean.numpy()[:, np.newaxis, np.newaxis]
            # b /= self.std.numpy()[:, np.newaxis, np.newaxis]
            if self.constants_data is not None:
                return_data[idx] = np.concatenate((self.constants_data, b), axis=0)
            else:
                return_data[idx] = copy.deepcopy(b)
            
        return return_data

    def load_data_process(self):
        while True:
            job_pid, idx, vname, height = self.index_queue.get()
            if job_pid not in self.compound_data_queue_dict:
                try:
                    self.lock.acquire()
                    for i in range(self.compound_data_queue_num):
                        if job_pid == self.arr[i]:
                            self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                            self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                            break
                    if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                        print("error", job_pid, self.arr)
                except Exception as err:
                    raise err
                finally:
                    self.lock.release()
            
            if vname in self.single_level_vnames:
                if idx < 0:
                    file = self.single_file_list[-idx]
                    # url = f"{single_acc_mean_dir}/{file[17:]}-{vname}.npy"
                    url = f"{single_acc_mean_dir}/{file[17:23]}{vname}.npy"
                    # print(url)
                else:
                    file = self.single_file_list[idx]
                    url = f"{self.data_dir}/{file}-{vname}.npy"
            elif vname in self.multi_level_vnames:
                if idx < 0:
                    file = self.file_list[-idx]
                    # url = f"{multi_acc_mean_dir}/{file[10:]}-{vname}-{height}.0.npy"
                    url = f"{multi_acc_mean_dir}/{file[10:16]}{vname}-{height}.0.npy"
                    # print(url)
                else:
                    file = self.file_list[idx]
                    url = f"{self.data_dir}/{file}-{vname}-{height}.0.npy"
            b = np.ndarray(self.a.shape, dtype=self.a.dtype, buffer=self.sharedmemory_dict[job_pid].buf)
            with io.BytesIO(self.client.get(url)) as f:
                try:
                    unit_data = np.load(f)
                    error = 0
                except Exception as err:
                    error = 1
                    print(f"{url}")
                    warnings.warn(f"{url}")
                    # raise ValueError(f"{url}")
            # unit_data = unit_data[np.newaxis, :, :]
            unit_data = self.normalization(vname, height, unit_data)
            if error != 1:
                if self.rm_equator:
                    b[self.index_dict1[(vname, height)], :360] = unit_data[:360]
                    b[self.index_dict1[(vname, height)], 360:] = unit_data[361:]
                else:
                    b[self.index_dict1[(vname, height)], :] = unit_data[:]
                del unit_data
            self.unit_data_queue.put((job_pid, idx, vname, height))

    def normalization(self, vname, height, data):
        if vname == "tp6h":
            data[data<0] = 0
            data = np.log(data+1)
        if vname in self.single_level_vnames:
            index = 0
        else:
            index = height_level.index(height)
        data -=  np.array(self.mean_std['mean'][vname][index], dtype=np.float32)
        data /= np.array(self.mean_std['std'][vname][index],dtype=np.float32)
        return data

    def __len__(self):
        data_len = len(self.file_list) - (self.length - 1) * self.sample_stride
        if self.split == 'valid':
            data_len -= self.pred_length * self.sample_stride + 1
            data_len = data_len // 2
        if data_len < 1:
            data_len = 1
        return data_len

    def get_meanstd(self):
        return_data_mean = []
        return_data_std = []
        
        for vname in self.single_level_vnames:
            return_data_mean.append(self.mean_std['mean'][vname])
            return_data_std.append(self.mean_std['std'][vname])
        for vname in self.multi_level_vnames:
            return_data_mean.append(self.mean_std['mean'][vname][self.height_level_indexes])
            return_data_std.append(self.mean_std['std'][vname][self.height_level_indexes])

        return torch.from_numpy(np.concatenate(return_data_mean, axis=0)[:, 0, 0]), torch.from_numpy(np.concatenate(return_data_std, axis=0)[:, 0, 0])

    def get_clim_daily(self, index):
        
        data_clim = self.get_data([-index])[-index]
        if self.constants_data is not None:
            data_clim = data_clim[self.constants_data.shape[0]:]

        return torch.from_numpy(data_clim).float()


    def __getitem__(self, index):
        index = min(index, len(self.file_list) - (self.length-1) * self.sample_stride - 1)
        if self.split == "valid":
            index = index * 2
        array_dict = self.get_data([index + i * self.sample_stride for i in range(self.length)])
        array_seq = [array_dict[index + i * self.sample_stride] for i in range(self.length)]
        del array_dict
        return array_seq

    def getitem(self, index):
        index = min(index, len(self.file_list) - (self.length-1) * self.sample_stride - 1)
        array_dict = self.get_data([index + i * self.sample_stride for i in range(self.length)])
        array_seq = [array_dict[index + i * self.sample_stride] for i in range(self.length)]
        del array_dict
        return array_seq
    

