# Frames Dataset Evaluation

The repository contains the code used to produce the evaluation of the Frametracking model
in [A Frame Tracking Model for Memory-Enhanced Dialogue Systems](https://arxiv.org/abs/1706.01690) 
by Hannes Schulz, Jeremie Zumer, Layla El Asri, and Shikhar Sharma.


## Installation

You should first download the Maluuba Frames dataset ([Link](https://www.microsoft.com/en-us/research/project/frames-dataset/#!download), [Backup Link](https://msropendata.com/datasets/1cc496ec-aaff-4576-b4bc-4a65798fa907)).

Clone the repository with
```bash
$ git clone https://github.com/Maluuba/frames
```
then install the package and its dependencies using

```bash
$ cd frames
$ pip install -U -e .
```

If you're inside a virtual environment or a conda environment, you can leave out the `-U`.


## Usage

To compute accuracy, you should use 

```bash
$ frametracking-evaluate eval FRAMES_JSON PREDICTION_JSON FOLD
```

where `FRAMES_JSON` is the file downloaded from the Maluuba website,
`PREDICTION_JSON` contains your predictions, and `FOLD` is an integer between 1
and 10 (inclusive). A fold is a split of the dataset corresponding to the
dialogues of a subset of the users. 

See below for how to [obtain the dialogues for a specific fold](#obtaining-train-test-split-for-each-fold).

Refer to the description below for the [required format of the predictions](#predictions-format).


### Finegrained Evaluation on Subtasks

The [frametracking-tagger](bin/frametracking-tagger) script allows you to tag turns in dialogues of the dataset, resulting in a more fine-grained evaluation.
An example [bash script](examples/label-frames-dataset.sh) is provided that tags some important categories.
See below on [how to use the frametracking-tagger](#dataset-filtering-inspection-and-tagging).

## Obtaining Train Test Split for Each Fold

The proposed evaluation scheme is to train on 9 of the folds and evaluate on the remaining one.
The folds are split up by user, so you can get the training and validation sets as follows:

```python
import json
from frames.utils import get_users_for_fold

with open("data/frames.json") as f:
    dialogues = json.load(f)

fold = 1
test_users = get_users_for_fold(fold)
train_users = get_users_for_fold(-fold)

train_dialogues = [d for d in dialogues if d['user_id'] in train_users]
test_dialogues = [d for d in dialogues if d['user_id'] in test_users]
```

You can get general info about a fold (number of dialogues, turns, and which
users are in it) by running

```bash
$ frametracking-evaluate foldinfo FRAMES_JSON
```

## Predictions Format

For every dialogue `d` turn `t`, `dialogue[d]['turns'][t]`, there is a field called `acts_without_labels`.
This contains a list of acts, e.g.

```
request(dst_city=NY)
```

in a JSON representation. Note that you can use `key_value_pairs()`
from [utils.py](frames/utils.py) to iterate over the JSON representation.

The [evaluation script](bin/frametracking-evaluate) expects a file which has the same dialogue/turn
structure, with an additional turn field `predictions`. This field has the same
acts and the same arguments in the same order as `acts_without_labels`, with
references added. There are two types of frame references, the slot-based and
the act-based ones. For the example above, the expected format would be:

```javascript
{'act': 'request',
 'args': [
   { 'key': 'ref',
     'val': [
       {'frame': XXXX,
        'annotations': [
          {'key': 'dst_city',
           'val': 'NY'}]}]},
   { 'key': 'ref',
     'val': [
      {'frame': YYYY}]}]}
```

The frames are the `F` frames from the previous turn (`t-1`),
`dialogues[i]['turns'][t-1]['frames']`, plus a potential new frame the user might have
created.
For turn 0, we assume frame 1 already exists. Thus,
* `XXXX` is list of floats, representing a multinomial distribution over the `F+1` frames described above, and
* `YYYY` is list of floats, representing one binomial distribution for every one of the `F+1` frames described above



## Dataset Filtering, Inspection and Tagging

The [tagging script](bin/frametracking-tagger) allows to select a subset of the turns or acts based on various criteria.
All criteria have to match for a turn/act to be counted/tagged.

For example, 

```bash
$ frametracking-tagger frames.json author user active-frame-changed has-act switch_frame prnt
```

prints all user acts where the active frame has been changed and a
`switch_frame` act is present. For all available filters, formatting options,
etc, see

```bash
frametracking-tagger --help   # and the documentation of the subcommands,
frametracking-tagger prnt --help   # etc.
```
