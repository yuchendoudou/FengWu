import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data.distributed import Sampler, Dataset
import torch.distributed as dist


T_co = TypeVar('T_co', covariant=True)



class Distributed_replay_Sampler(Sampler[T_co]):

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        if hasattr(self.dataset, "replay_buff_rate"):
            self.replay_buff_rate = self.dataset.replay_buff_rate
        else:
            self.replay_buff_rate = 0

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank*self.total_size//self.num_replicas:(self.rank+1)*self.total_size//self.num_replicas]

        assert len(indices) == self.num_samples

        if self.epoch == 0:
            out_indices = indices[:50]
            for i in range(50, len(indices)):
                out_indices.append(indices[i])
                out_indices = out_indices + [-1 for j in range(self.replay_buff_rate)]
        else:
            out_indices = []
            for i in range(len(indices)):
                out_indices.append(indices[i])
                out_indices = out_indices + [-1 for j in range(self.replay_buff_rate)]


        return iter(out_indices)

    def __len__(self) -> int:
        if self.epoch == 0:
            return self.num_samples + (self.num_samples - 50) * self.replay_buff_rate
        else:
            return self.num_samples * (self.replay_buff_rate + 1)


    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch




