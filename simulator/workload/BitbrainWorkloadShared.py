"""Share immutable source sequences without changing protocol016 task behavior."""
from .BitbrainWorkloadAdapted import AdaptedBWGD2


class SharedAdaptedBWGD2(AdaptedBWGD2):
    def __init__(self, mean, sigma, replay_seed):
        self.sequence_pool={}
        super().__init__(mean,sigma,replay_seed)

    def share(self, values):
        key=tuple(values)
        return self.sequence_pool.setdefault(key,key)

    def adapt_new_tasks(self, first):
        super().adapt_new_tasks(first)
        for _,_,ips,ram,_ in self.createdContainers[first:]:
            ips.ips_list=self.share(ips.ips_list)
            for name in ('size_list','read_list','write_list'):
                setattr(ram,name,self.share(getattr(ram,name)))
