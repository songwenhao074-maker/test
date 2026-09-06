"""Acceptance checks for online causality, frozen weights, topology and resume."""
import io
import unittest
import numpy as np
import torch
from analyze_ftmoe_online import summarize_arrays, bootstrap_difference
from run_ftmoe_online import ROOT, OnlineSession, Replay, resolve_checkpoint, tolerance_label
from recovery.PreGANSrc.src.ftmoe_online import OnlineFTMoE
from train_ftmoe_end_to_end import create_model


class OnlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(3)
        torch.set_num_interop_threads(1)
        cls.checkpoint,_=resolve_checkpoint(1)
        data=ROOT/'artifacts/ftmoe_end_to_end/data/protocol_004_physical'
        cls.host=np.load(data/'time_series.npy')[5].reshape(-1,16,7)
        cls.graph=np.load(data/'container_demand_series.npy')[5].reshape(-1,16,7)
        cls.schedule=np.load(data/'schedule_series.npy')[5]

    def replay(self,n=110):
        # Small artificial labels are exclusively unit-test fixtures, never experiment streams.
        labels=np.zeros((n+1,16),np.int64)
        labels[::3,0]=1;labels[1::4,1]=2;labels[2::5,2]=3
        a={'host_features':self.host[:n+1].copy(),'demands':self.graph[:n+1].copy(),
           'schedules':self.schedule[:n+1].copy(),'raw_labels':labels,
           'capacities':np.array([[4029,4295 if h<8 else 8192,32212] for h in range(16)],np.float32)}
        return Replay(a,self.checkpoint['normalization'],n)

    def session(self,method='D',n=110):
        return OnlineSession(self.checkpoint,method,1,self.replay(n),0. if method=='A' else 3e-5,301)

    def test_legacy_conversion_and_equal_initial_states(self):
        old=create_model('v4',1,self.checkpoint['normalization']);old.load_state_dict(self.checkpoint['model']);old.eval()
        x,s,g=self.replay().window(15)
        with torch.no_grad():reference=old(x[None],s[None],g[None])
        hashes=[]
        for method in 'ABCD':
            online=OnlineFTMoE(self.checkpoint,method,1)
            with torch.no_grad():actual=online(x[None],s[None],g[None])
            for key in ('detection_logits','class_logits'):
                torch.testing.assert_close(actual[key],reference[key],rtol=0,atol=0)
            hashes.append(self.session(method).initial_state_hash)
        self.assertEqual(len(set(hashes)),1)

    def test_frozen_sets_optimizer_and_statistics(self):
        for method in 'ABCD':
            session=self.session(method,30);initial=session.model.state_hash()
            for _ in range(20):session.step()
            self.assertEqual(session.model.frozen_hash(),session.initial_frozen_hash)
            if method=='A':self.assertEqual(initial,session.model.state_hash())
            else:
                self.assertNotEqual(initial,session.model.state_hash())
                self.assertEqual({id(p) for p in session.model.parameters() if p.requires_grad},
                    {id(p) for group in session.optimizer.param_groups for p in group['params']})
            self.assertEqual(session.model.eagate.routing_samples,20*16)
            before=dict(session.model.eagate.activation_counts)
            x,s,g=session.replay.window(0)
            with torch.no_grad():session.model(x[None],s[None],g[None])
            self.assertEqual(before,session.model.eagate.activation_counts)

    def test_future_label_isolation(self):
        first=self.session('B',35);second=self.session('B',35)
        second.replay.arrays['raw_labels'][20:]=3
        for _ in range(20):first.step();second.step()
        np.testing.assert_array_equal(first.predictions['probability'][:20],second.predictions['probability'][:20])
        self.assertEqual(first.model.state_hash(),second.model.state_hash())
        self.assertLessEqual(max(first.buffer),18)
        for _ in range(10):first.step();second.step()
        self.assertNotEqual(first.model.state_hash(),second.model.state_hash())

    def test_tolerance_delay_and_tie(self):
        raw=np.zeros((5,16),np.int64);raw[1,0]=2;raw[3,0]=3
        with self.assertRaises(ValueError):tolerance_label(raw,2,2)
        self.assertEqual(tolerance_label(raw,2,3)[0],2)
        raw[2,0]=1;self.assertEqual(tolerance_label(raw,2,3)[0],1)
        self.assertEqual(tolerance_label(raw,0,1)[0],2)

    def test_dynamic_topology_preserves_adam_and_limits(self):
        session=self.session('D',30)
        for _ in range(20):session.step()
        gate=session.model.eagate
        survivor=gate.key_rows['2'];moments={k:v.clone() if torch.is_tensor(v) else v for k,v in session.optimizer.state[survivor].items()}
        gate.activation_counts={'0':0,'1':0,'2':10,'3':10};gate.routing_samples=20
        gate.unmatched_count=1;gate.unmatched_vectors.append(torch.ones(64))
        event=session.model.adapt(session.optimizer)
        self.assertEqual(event['removed'],['0','1']);self.assertEqual(event['added'],['4'])
        self.assertIs(survivor,gate.key_rows['2'])
        for key,value in moments.items():torch.testing.assert_close(session.optimizer.state[survivor][key],value,rtol=0,atol=0)
        self.assertNotIn(gate.key_rows['4'],session.optimizer.state)
        self.assertEqual(float(gate.threshold_rows['4']),0.)
        self.assertEqual(float(gate.experts['4'][-1].weight.abs().sum()),0.)
        for _ in range(10):
            gate.activation_counts={key:1 for key in gate.ids};gate.routing_samples=1
            gate.unmatched_count=1;gate.unmatched_vectors.append(torch.ones(64))
            session.model.adapt(session.optimizer)
        self.assertEqual(len(gate.ids),8)
        x,s,g=session.replay.window(0);output=session.model.predict_online(x[None],s[None],g[None])
        self.assertTrue((output['active_experts']>=1).all());self.assertTrue((output['active_experts']<=4).all())
        self.assertTrue(torch.isfinite(output['detection_logits']).all())
        for _ in range(10):session.step()
        self.assertIn(gate.key_rows['4'],session.optimizer.state)

    def test_resume_after_topology_event(self):
        continuous=self.session('D',115)
        for _ in range(99):continuous.step()
        continuous.model.eagate.unmatched_count+=1
        continuous.model.eagate.unmatched_vectors.append(torch.ones(64))
        for _ in range(6):continuous.step()
        self.assertGreater(continuous.model.eagate.next_id,4)
        blob=io.BytesIO();torch.save(continuous.save(),blob);blob.seek(0)
        resumed=self.session('D',115);resumed.restore(torch.load(blob,weights_only=False))
        while continuous.cursor<115:continuous.step();resumed.step()
        continuous.finish();resumed.finish()
        self.assertEqual(continuous.model.state_hash(),resumed.model.state_hash())
        for key in continuous.predictions:
            if key!='prediction_seconds':np.testing.assert_array_equal(continuous.predictions[key],resumed.predictions[key])
        self.assertEqual(continuous.model.eagate.ids,resumed.model.eagate.ids)
        self.assertEqual(list(continuous.buffer),list(resumed.buffer))

    def test_reference_does_not_change_capacity_or_routing(self):
        session=self.session('D',20)
        for _ in range(10):session.step()
        x,s,g=session.replay.window(0)
        blocks={'fixture':(x[None],g[None],s[None],torch.zeros((1,16),dtype=torch.long))}
        before=session.model.state_hash();counts=dict(session.model.eagate.activation_counts)
        session.evaluate_reference(blocks)
        self.assertEqual(before,session.model.state_hash());self.assertEqual(counts,session.model.eagate.activation_counts)

    def test_metrics_merge_and_empty_windows(self):
        p=np.zeros((200,16),np.float32);c=np.full((200,16,3),1/3,np.float32);y=np.zeros((200,16),np.int64)
        y[150:,0]=1;p[150:175,0]=1;p[100:110,1]=1
        result=summarize_arrays(p,c,y,y)
        self.assertIsNone(result['first_half']['f1'])
        self.assertAlmostEqual(result['second_half']['f1'],50/(50+10+25))
        self.assertEqual(result['second_half']['fp'],10)
        self.assertIsNone(result['rolling'][0]['f1'])
        self.assertEqual(result['rolling'][0]['positives'],0)
        self.assertEqual(result['second_half']['samples'],1600)

    def test_bootstrap_uses_crossed_pairs(self):
        rows=[]
        for s in (1,2):
            for r in (301,302):
                for method,value in [('B',.7),('D',.8)]:
                    rows.append({'configuration':{'model_seed':s,'replay_seed':r,'method':method},'metrics':{'second_half':{'f1':value}}})
        result=bootstrap_difference(rows,'D','B')
        np.testing.assert_allclose(result['ci95_paired_crossed_bootstrap'],[.1,.1])


if __name__=='__main__':unittest.main(verbosity=2)
