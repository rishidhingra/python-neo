# -*- coding: utf-8 -*-
"""
neo.io have been split in 2 level API:
  * neo.io: this API give neo object
  * neo.rawio: this API give raw data as they are in files.

Developper are encourage to use neo.rawio.

When this is done the neo.io is done automagically with
this king of following code.

Author: sgarcia

"""

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.oebinrawio import OEBinRawIO


class OEBinIO(ExampleRawIO, BaseFromRaw):
    name = 'OEBin IO'
    description = "Open Ephys Binary IO"

    # This is an inportant choice when there are several channels.
    #   'split-all' :  1 AnalogSignal each 1 channel
    #   'group-by-same-units' : one 2D AnalogSignal for each group of channel with same units
    _prefered_signal_group_mode = 'group-by-same-units'

    def __init__(self, dirname=''):
        OEBinRawIO.__init__(self, dirname=dirname)
        BaseFromRaw.__init__(self, filename)
