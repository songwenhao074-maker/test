"""Focused checks for the new physical demand adapter."""
import copy
import json
from types import SimpleNamespace
import unittest
import numpy as np
from simulator.workload.BitbrainWorkloadAdapted import AdaptedBWGD2, TrainingMarkovDisk, LAW_PATH, map_cpu
from simulator.container.IPSModels.IPSMBitbrain import IPSMBitbrain
from simulator.container.RAMModels.RMBitbrain import RMBitbrain


class AdapterTests(unittest.TestCase):
    def test_work_budget_and_retry_are_not_double_scaled(self):
        w=AdaptedBWGD2.__new__(AdaptedBWGD2)
        w.replay_seed=301;w.disk_law=json.loads(LAW_PATH.read_text())
        cpu=IPSMBitbrain([0,1,1000,4000],5000,4,20)
        ram=RMBitbrain([100]*4,[9]*4,[8]*4)
        w.createdContainers=[(0,0,cpu,ram,None)]
        w.adapt_new_tasks(0)
        np.testing.assert_array_equal(cpu.ips_list,[0,2,1000,1860])
        self.assertEqual(cpu.max_ips,1860)
        self.assertEqual(ram.size_list,[200]*4)
        w.adapt_new_tasks(1)
        self.assertEqual(ram.size_list,[200]*4)
        cpu.container=SimpleNamespace(env=SimpleNamespace(interval=1,intervaltime=300),startAt=0)
        self.assertEqual(cpu.getIPS(),2)
        self.assertEqual(cpu.totalInstructions,2862*300)
        cpu.completedInstructions=cpu.totalInstructions
        self.assertEqual(cpu.getIPS(),0)

    def test_disk_identity_idempotence_and_restore(self):
        law=json.loads(LAW_PATH.read_text())
        a=TrainingMarkovDisk(law,301,7);b=TrainingMarkovDisk(law,301,7)
        env=SimpleNamespace(interval=0)
        for disk in (a,b):disk.allocContainer(SimpleNamespace(env=env,startAt=0,hostid=0))
        for t in range(15):
            env.interval=t
            self.assertEqual(a.disk(),a.disk())
            self.assertEqual(a.disk(),b.disk())
        saved=a.save();results=[]
        for t in range(15,80):
            env.interval=t;results.append(a.disk())
        a.restore(saved);a.container.hostid=12
        for t,expected in zip(range(15,80),results):
            env.interval=t;self.assertEqual(a.disk(),expected)
        self.assertTrue(all(0<=v[0]<=9000 for v in results))

    def test_transition_fit_has_no_block_boundary(self):
        law=json.loads(LAW_PATH.read_text())
        counts=np.asarray(law['transition_counts'])
        self.assertEqual(int(counts.sum()),5*201*16)
        np.testing.assert_allclose(np.asarray(law['transition']).sum(1),1)
        self.assertEqual(law['train_blocks'],[0,1,2,3,4])
        np.testing.assert_array_equal(map_cpu([0,1,1900]),[0,2,1860])


if __name__=='__main__':unittest.main()
