# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import logging
import re
from itertools import zip_longest
import numpy as np
import tabulate
import colorama


EPSILON = 1e-6


def key_value_pairs(act, with_frames=False, current_frame=None,
                    shuffle_slots=False, rng=None):
    """
    iterates over all key value pairs in an act. If with_frames is true,
    returns triples (frame_id, key, value). If a key value pair does not have a
    frame_id, yield frame_id=current_frame. If a ref without annotations is
    given, yields (frame_id,"NOANNO,None). Unannotated refs are produced
    ordered by frame_id.
    """
    unannotated_refs = []
    has_refs = False
    if shuffle_slots:
        rng.shuffle(act['args'])
    for arg in act['args']:
        if arg['key'] in ('ref', 'read', 'write'):
            for frame in arg['val']:
                for kv in frame.get('annotations', []):
                    has_refs = True
                    if kv['key'] in ('ref', 'read', 'write'):
                        print(kv, frame)
                        assert False
                    if with_frames:
                        yield frame['frame'], kv['key'], kv.get('val', None)
                    else:
                        yield kv['key'], kv.get('val', None)
                if not frame.get('annotations', []) and with_frames:
                    unannotated_refs += (frame['frame'], "NOANNO", None),
        elif arg['key'] != 'id':
            if with_frames:
                yield current_frame, arg['key'], arg.get('val', None)
            else:
                yield arg['key'], arg.get('val', None)
    if unannotated_refs:
        for ua in sorted(unannotated_refs):
            yield ua
    elif not has_refs:
        if with_frames:
            yield current_frame, "NOANNO", None


class ActJSON2Str(object):
    """
    Compact string representation of a dialogue act in JSON format.
    """
    def __init__(self, act_sep=',', arg_sep=','):
        self.act_sep = act_sep
        self.arg_sep = arg_sep

    def transform_val(self, val):
        if isinstance(val, bool):
            return 'T' if val else 'F'
        if val is None:
            return val
        if not isinstance(val, str):
            val = str(val)
        val = re.sub(r'([,\)\}])', r'\\\1', val)
        return val

    @classmethod
    def color_if_max(cls, val, idx, argmaxidx):
        val = ("%1.2f" % val).rstrip('0').rstrip('.')
        if idx == argmaxidx:
            val = colorama.Fore.GREEN + val + colorama.Fore.RESET
        return val

    def transform_refs(self, frame, annotations):
        if isinstance(frame, list):
            am = np.argmax(frame)
            frame = "[" + ",".join(self.color_if_max(f, idx, am)
                                   for idx, f in enumerate(frame)) + "]"
        else:
            frame = "%d" % frame
        if len(annotations) == 0:
            return '%s' % frame
        return '%s{%s}' % (frame, self.arg_sep.join(
            self.transform_arg(anno, norefs=True)
            for anno in annotations))

    def transform_arg(self, arg, norefs=False):
        key = arg['key']
        if not norefs and key in 'read write ref'.split():
            return key + '=[' + ";".join(
                self.transform_refs(r['frame'], r['annotations'])
                for r in arg['val']) + "]"
        val = self.transform_val(arg.get('val', None))
        if val is None:
            return key
        return key + '=' + val

    def transform_one(self, jsonact):
        name = jsonact['name']
        args = jsonact['args']
        args = [self.transform_arg(arg) for arg in args]
        return name + "(" + self.arg_sep.join(args) + ")"

    def transform(self, jsonacts):
        """
        takes a list of JSON objects representing acts, returns a list of
        strings, each representing an act.
        """
        return [self.transform_one(act) for act in jsonacts]


def get_users_for_fold(fold):
    folds = {'U21E41CQP': 1,
             'U23KPC9QV': 1,
             'U21RP4FCY': 2,
             'U22HTHYNP': 3,
             'U22K1SX9N': 4,
             'U231PNNA3': 5,
             'U23KR88NT': 6,
             'U24V2QUKC': 7,
             'U260BGVS6': 8,
             'U2709166N': 9,
             'U2AMZ8TLK': 10}

    if fold < 0:
        ret = [k for k, v in folds.items() if v != -fold]
    else:
        ret = [k for k, v in folds.items() if v == fold]
    return ret


def make_table(cnt, cnt_noanno, cnt_comb):
    ret = [["Key", "count", "accuracy (%)", "loglik", "with-noanno (%)", "with-noanno (loglik)"]]
    ret2 = [["Key", "count", "accuracy (%)", "loglik"]]

    keys = {k.rsplit("_", 1)[0] for k in cnt.keys()}

    for k in sorted(keys):
        n = cnt[k + '_cnt']
        ret.append([k, n, "%2.1f" % (100. * cnt[k + "_accuracy"] / n),
                    "%2.1f" % (cnt[k + "_loglik"] / n)])

        n = cnt_noanno[k + '_cnt']
        n = max(1, n)
        ret[-1].extend(("%2.1f" % (100. * cnt_noanno[k + "_accuracy"] / n),
                        "%2.1f" % (cnt_noanno[k + "_loglik"] / n)))
    keys2 = {k.rsplit("_", 1)[0] for k in cnt_comb.keys()}
    keys2 = sorted(keys2, key=lambda a: cnt_comb[a + '_count'])
    for key in keys2:
        ret2.append([key, cnt_comb[key + "_count"],
                     "%2.1f" % (100. * cnt_comb[key + "_accuracy"] / cnt_comb[key + "_count"]),
                     "%2.1f" % (cnt_comb[key + "_loglik"] / cnt_comb[key + "_count"])])
    return tabulate.tabulate(ret, headers='firstrow'), tabulate.tabulate(ret2, headers='firstrow')


def cmp_turns(cnt, cnt_noanno, cnt_comb, gt_turn, pred_turn):
    """
    input format of gt/pred is the same, except that in pred has -- instead of an
    int 'frame' -- an array containing the distribution over the frames.
    """
    gt_acts = gt_turn['labels']['acts']
    if 'predictions' not in pred_turn and pred_turn['author'] != 'wizard':
        raise RuntimeError('Predictions missing from user turn', list(pred_turn.keys()), pred_turn['text'])
    pred_acts = pred_turn['predictions']['acts'] if 'predictions' in pred_turn else {}
    if len(pred_acts) != len(gt_acts):
        raise RuntimeError("Different number of acts in a turn!")

    current_frame = gt_turn['labels']['active_frame']
    for gt_act, pred_act in zip(gt_acts, pred_acts):
        gt_slotval_iter = key_value_pairs(gt_act, with_frames=True,
                                          current_frame=current_frame)
        pred_slotval_iter = key_value_pairs(pred_act, with_frames=True)

        gt_noanno = []
        pred_noanno = []

        corrects, noanno_corrects = [], []
        logliks, noanno_logliks = [], []
        act_list, slot_list = [], []

        for gt, pred in zip_longest(gt_slotval_iter, pred_slotval_iter):
            found_noanno = False
            if gt is not None and gt[1] == 'NOANNO':
                gt_noanno.append(gt[0])
                found_noanno = True

            if pred is not None and pred[1] == 'NOANNO' and pred[0] is not None:
                pred_noanno = pred[0]  # there should be exactly one NOANNO in the prediction
                found_noanno = True

            if found_noanno:
                continue

            if gt is None or pred is None or gt[1] != pred[1]:
                raise RuntimeError("Slot types do not match: %s, %s", gt[1], pred[1])

            if gt[2] != pred[2]:
                logging.warning("Slot values do not match: %s, %s", gt, pred)

            # accuracy
            correct = gt[0] == (np.argmax(pred[0])) + 1

            if not correct:
                pred_turn['labels']['tags'] = list(set(pred_turn['labels'].get('tags', [])).union(set(['wrong'])))

            # average log likelihood of correct solution
            pred_sum = sum(pred[0]) if pred[0] is not None else 0
            if pred_sum > 1:
                pred = list(pred)
                if pred_sum > 1.00001:
                    logging.warning("Unnormalized distribution (sum=%2.5f), dividing by sum., %s", pred_sum, gt_turn['text'])
                pred[0] = np.asarray([p / pred_sum for p in pred[0]]) if pred[0] is not None else None

            assert gt[0] > 0
            try:
                loglik = np.log(max(EPSILON, pred[0][gt[0] - 1]) if pred[0] is not None else EPSILON)
            except IndexError:
                loglik = np.log(EPSILON)
            corrects.append(correct)
            logliks.append(loglik)
            act_list.append(gt_act['name'])
            slot_list.append(gt[1])

            if isinstance(pred_noanno, int):
                pred_noanno = [pred_noanno]

            if len(pred_noanno) > 0 and not (0 <= max(pred_noanno) <= 1):
                raise RuntimeError("Unnormalized distribution (value=%2.5f)", max(pred_noanno))

        if len(gt_noanno) == 0:
            if len(pred_noanno) > 0:
                noanno_corrects.append(max(pred_noanno) < 0.5)
                noanno_logliks.append(np.log(np.maximum(EPSILON, pred_noanno)).mean())
        else:
            ok = (set(np.where(np.asarray(pred_noanno) > 0.5)[0] + 1) == set(gt_noanno))
            noanno_corrects.append(ok)
            other_ind = [i for i in range(0, len(pred_noanno)) if i + 1 not in gt_noanno]
            try:
                noanno_logliks.append(
                    np.concatenate(
                        (np.log(1 - np.take(pred_noanno, [i - 1 for i in gt_noanno])),
                         np.log(np.take(pred_noanno, other_ind)))).mean())
            except IndexError:
                noanno_logliks.append(np.log(EPSILON))

        for correct, loglik, act, slot in zip(corrects, logliks, act_list, slot_list):
            for tag in gt_turn['labels'].get('tags', []) + ['total']:
                cnt[tag + "_cnt"] += 1
                cnt[tag + "_accuracy"] += correct
                cnt[tag + '_loglik'] += loglik
            cnt_comb[act + ':' + slot + "_accuracy"] += correct
            cnt_comb[act + ':' + slot + "_loglik"] += loglik
            cnt_comb[act + ':' + slot + '_count'] += 1

        corrects, logliks = [], []

        for correct, loglik in zip(noanno_corrects, noanno_logliks):
            for tag in gt_turn['labels'].get('tags', []) + ['total']:
                cnt_noanno[tag + "_cnt"] += 1
                cnt_noanno[tag + "_accuracy"] += correct
                cnt_noanno[tag + '_loglik'] += loglik
