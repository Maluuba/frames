#!/bin/sh

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

I=data/frames.json
O=data/tagged.json
FREX=frametracking-tagger

mkdir -p data
if [ ! -e $I ]; then
	echo "Please download $I from https://www.microsoft.com/en-us/research/project/frames-dataset/#!download"
	exit 1
fi


# frame changed, using switch_frame
python $FREX --saveas $O $I author user active-frame-changed has-act switch_frame tag --clear-first framechange.switch_frame
python $FREX --saveas $O $O author user active-frame-changed has-act switch_frame prev-has-act 'offer suggest' tag framechange.switch_frame.offer
python $FREX --saveas $O $O author user active-frame-changed has-act switch_frame prev-has-no-act 'offer suggest' tag framechange.switch_frame.nooffer
python $FREX --saveas $O $O author user active-frame-changed has-act switch_frame has-act --allowed-slot-args '' switch_frame tag framechange.switch_frame.noargs

python $FREX --saveas $O $O author user has-act request_compare tag reqcomp

# frame changed with value
python $FREX --saveas $O $O author user active-frame-changed has-act -i switch_frame tag framechange.valuechange

# noframechange
python $FREX --saveas $O $O author user active-frame-changed -i has-act -i switch_frame tag noframechange

# noframechange, with preceding offer
python $FREX --saveas $O $O author user active-frame-changed -i prev-has-act 'offer suggest' tag noframechange.offer

# noframechange, without preceding offer
python $FREX --saveas $O $O author user active-frame-changed -i prev-has-no-act 'offer suggest' tag noframechange.nooffer

python $FREX --saveas $O $O author user has-noanno tag noanno
python $FREX --saveas $O $O author user has-noanno -i tag no_noanno
