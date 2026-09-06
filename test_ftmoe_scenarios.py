"""Check source-time causality, resource units, and missing-data handling."""
import unittest
from types import SimpleNamespace
import numpy as np
from simulator.workload.Google2011Workload import GlobalCPU,GlobalRAM,GlobalDisk
from prepare_google2011_pilot import add_measurement,fill_single_gaps,ORIGIN,WIDTH,BINS


class ScenarioTests(unittest.TestCase):
    def test_sequence_interning_preserves_values(self):
        from simulator.workload.BitbrainWorkloadShared import SharedAdaptedBWGD2
        workload=SharedAdaptedBWGD2.__new__(SharedAdaptedBWGD2)
        workload.sequence_pool={}
        first=workload.share([0.,2.,1860.])
        second=workload.share([0.,2.,1860.])
        self.assertIs(first,second)
        self.assertEqual(first,(0.,2.,1860.))
        self.assertIsNot(first,workload.share([0.,2.,1859.]))

    def test_global_time_and_exhaustion(self):
        values=np.array([[1.,2.,3.],[4.,5.,6.]])
        container=SimpleNamespace(env=SimpleNamespace(interval=1),startAt=50)
        models=[GlobalCPU(values),GlobalRAM(values),GlobalDisk(values)]
        for m in models:m.allocContainer(container)
        self.assertEqual(models[0].getIPS(),1.)
        container.env.interval=2
        self.assertEqual(models[1].ram(),(5.,1.,1.))
        self.assertEqual(models[2].disk(),(6.,1.,1.))
        container.env.interval=3
        with self.assertRaises(IndexError):models[0].getIPS()

    def test_overlap_weighting(self):
        sums=np.zeros((1,BINS,3));weights=np.zeros((1,BINS))
        add_measurement(sums,weights,[ORIGIN,ORIGIN+WIDTH//2,1,0,2.,4.,6.],0)
        add_measurement(sums,weights,[ORIGIN+WIDTH//2,ORIGIN+WIDTH,1,0,4.,6.,8.],0)
        np.testing.assert_allclose(sums[0,0]/weights[0,0],[3.,5.,7.])
        self.assertEqual(weights[0,0],WIDTH)

    def test_only_isolated_missing_bins_filled(self):
        values=np.arange(21.).reshape(7,3)
        coverage=np.array([True,False,True,False,False,True,False])
        fixed,valid=fill_single_gaps(values,coverage)
        self.assertEqual(valid.tolist(),[True,True,True,False,False,True,False])
        np.testing.assert_allclose(fixed[1],values[0])
        changed=values.copy();changed[2:]+=1000
        changed_fixed,_=fill_single_gaps(changed,coverage)
        np.testing.assert_array_equal(fixed[:2],changed_fixed[:2])


if __name__=='__main__':unittest.main()
