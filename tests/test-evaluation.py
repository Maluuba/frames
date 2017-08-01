from collections import defaultdict
import numpy as np
import copy
from frames.utils import EPSILON, cmp_turns


MOCK_FRAME = dict()


MOCK_DATASET = [
    dict(id='dialogue-one',
         user_id='U21E41CQP',   # in fold 1
         turns=[
             dict(author='user',
                  text='foo bar',
                  labels=dict(active_frame=2,
                              acts=[],
                              acts_without_refs=[],
                              frames=[MOCK_FRAME] * 3)),
             dict(
                 author='user',
                 text='foo bar',
                 labels=dict(
                     frames=[MOCK_FRAME] * 4,
                     active_frame=2,
                     tags=['A', 'A.B', 'C'],
                     acts=[
                         dict(name='inform',
                              args=[dict(key='dst_city', val='New York'),
                                    dict(key='ref', val=[
                                        dict(frame=4,   # a new frame!
                                             annotations=[
                                                 dict(key='duration',
                                                      val='14')])])]),
                         dict(name='request',
                              args=[dict(key='or_city'),
                                    dict(key='ref', val=[dict(frame=2)])])],  # an old frame, but no annotations.
                     acts_without_labels=dict(
                         name='inform',
                         args=[dict(key='dst_city', val='New York'),
                               dict(key='duration', val='14')]))),

             dict(
                 author='user',
                 text='foo bar',
                 labels=dict(
                     active_frame=2,
                     tags=['C'],
                     acts=[
                         dict(name='inform',
                              args=[dict(key='dst_city', val='New York'),
                                    dict(key='ref', val=[
                                        dict(frame=4,   # now it's not new anymore
                                             annotations=[
                                                 dict(key='duration',
                                                      val='14')])])]),
                         dict(name='request',
                              args=[dict(key='or_city'),
                                    dict(key='ref', val=[dict(frame=2)])])],  # an old frame, but no annotations.
                     acts_without_labels=dict(
                         name='inform',
                         args=[dict(key='dst_city', val='New York'),
                               dict(key='duration', val='14')])))
         ])
]


def make_mock_predictions():
    global MOCK_PREDICTIONS
    MOCK_PREDICTIONS = copy.deepcopy(MOCK_DATASET)
    for dlg in MOCK_PREDICTIONS:
        for tidx, turn in enumerate(dlg['turns']):
            turn['predictions'] = copy.deepcopy(turn['labels'])
            n_pred_frames = len(dlg['turns'][tidx - 1]['labels']['frames']) if tidx > 0 else 0

            # create a dummy softmax distribution
            eye = np.eye(n_pred_frames + 1)  # +1: new frame
            eye = np.clip(eye, EPSILON, 1. - EPSILON)
            eye /= eye.sum(-1, keepdims=True)

            # translate ground truth refs into predictions
            for act in turn['predictions']['acts']:
                # new arguments for this act (with at least a slot-type, and maybe a value)
                new_args = []

                # new arguments for this act (just a frame reference without slot/value)
                noanno = np.ones(n_pred_frames + 1) * EPSILON
                for arg in act['args']:
                    if arg['key'] != 'ref':
                        # an implicit reference to the current frame
                        new_args.append(dict(key='ref',
                                             val=[dict(frame=eye[turn['labels']['active_frame'] - 1],
                                                       annotations=[arg])]))
                    else:
                        # an explicit reference to a frame
                        new_val = []
                        for ref in arg['val']:
                            if ref.get('annotations', []):
                                new_val.append(dict(frame=eye[ref['frame'] - 1],  # frames start at 1
                                                    annotations=ref['annotations']))
                            else:
                                noanno[ref['frame'] - 1] = 1. - EPSILON
                        arg['val'] = new_val
                        new_args.append(arg)
                new_args.append(dict(key='ref',
                                     val=[dict(frame=noanno)]))
                act['args'] = new_args


def test_cmp_turns():
    from nose.tools import eq_
    cnt, cnt_noanno, cnt_comb = (defaultdict(float) for i in range(3))
    make_mock_predictions()

    cmp_turns(cnt, cnt_noanno, cnt_comb, MOCK_DATASET[0]['turns'][1], MOCK_PREDICTIONS[0]['turns'][1])
    eq_(cnt['total_cnt'], 3)
    eq_(cnt['total_accuracy'], 3)
    eq_(cnt['A_cnt'], 3)
    eq_(cnt['A.B_cnt'], 3)
    eq_(cnt['C_cnt'], 3)

    eq_(cnt_noanno['total_cnt'], 2)
    eq_(cnt_noanno['total_accuracy'], 2)
    eq_(cnt_noanno['A_cnt'], 2)
    eq_(cnt_noanno['A.B_cnt'], 2)
    eq_(cnt_noanno['C_cnt'], 2)
    cmp_turns(cnt, cnt_noanno, cnt_comb, MOCK_DATASET[0]['turns'][2], MOCK_PREDICTIONS[0]['turns'][2])
    eq_(cnt['total_cnt'], 6)
    eq_(cnt['total_accuracy'], 6)
    eq_(cnt['A_cnt'], 3)
    eq_(cnt['A.B_cnt'], 3)
    eq_(cnt['C_cnt'], 6)
    eq_(cnt_noanno['total_cnt'], 4)
    eq_(cnt_noanno['total_accuracy'], 4)
    eq_(cnt_noanno['A_cnt'], 2)
    eq_(cnt_noanno['A.B_cnt'], 2)
    eq_(cnt_noanno['C_cnt'], 4)


if __name__ == '__main__':
    test_cmp_turns()
