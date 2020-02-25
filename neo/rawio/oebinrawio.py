# -*- coding: utf-8 -*-
"""
ExampleRawIO is a class of a  fake example.
This is to be used when coding a new RawIO.


Rules for creating a new class:
  1. Step 1: Create the main class
    * Create a file in **neo/rawio/** that endith with "rawio.py"
    * Create the class that inherits BaseRawIO
    * copy/paste all methods that need to be implemented.
      See the end a neo.rawio.baserawio.BaseRawIO
    * code hard! The main difficulty **is _parse_header()**.
      In short you have a create a mandatory dict than
      contains channel informations::

            self.header = {}
            self.header['nb_block'] = 2
            self.header['nb_segment'] = [2, 3]
            self.header['signal_channels'] = sig_channels
            self.header['unit_channels'] = unit_channels
            self.header['event_channels'] = event_channels

  2. Step 2: RawIO test:
    * create a file in neo/rawio/tests with the same name with "test_" prefix
    * copy paste neo/rawio/tests/test_examplerawio.py and do the same

  3. Step 3 : Create the neo.io class with the wrapper
    * Create a file in neo/io/ that endith with "io.py"
    * Create a that inherits both your RawIO class and BaseFromRaw class
    * copy/paste from neo/io/exampleio.py

  4.Step 4 : IO test
    * create a file in neo/test/iotest with the same previous name with "test_"
prefix
    * copy/paste from neo/test/iotest/test_exampleio.py



"""
from __future__ import unicode_literals, print_function, division, \
    absolute_import

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)

import os
import json

import numpy as np


class OEBinRawIO(BaseRawIO):
    """
    Class for "reading" fake data from an imaginary file.

    For the user, it give acces to raw data (signals, event, spikes) as they
    are in the (fake) file int16 and int64.

    For a developer, it is just an example showing guidelines for someone who wants
    to develop a new IO module.

    Two rules for developers:
      * Respect the :ref:`neo_rawio_API`
      * Follow the :ref:`io_guiline`

    This fake IO:
        * have 2 blocks
        * blocks have 2 and 3 segments
        * have 16 signal_channel sample_rate = 10000
        * have 3 unit_channel
        * have 2 event channel: one have *type=event*, the other have
          *type=epoch*


    Usage:
        >>> import neo.rawio
        >>> r = neo.rawio.ExampleRawIO(filename='itisafake.nof')
        >>> r.parse_header()
        >>> print(r)
        >>> raw_chunk = r.get_analogsignal_chunk(block_index=0, seg_index=0,
                            i_start=0, i_stop=1024,  channel_names=channel_names)
        >>> float_chunk = reader.rescale_signal_raw_to_float(raw_chunk, dtype='float64',
                            channel_indexes=[0, 3, 6])
        >>> spike_timestamp = reader.spike_timestamps(unit_index=0, t_start=None, t_stop=None)
        >>> spike_times = reader.rescale_spike_timestamp(spike_timestamp, 'float64')
        >>> ev_timestamps, _, ev_labels = reader.event_timestamps(event_channel_index=0)

    """
    extensions = []
    rawmode = 'one-dir'

    def __init__(self, dirname=''):
        BaseRawIO.__init__(self)
        # note that this filename is ued in self._source_name
        self.dirname = dirname

    def _source_name(self):
        # this function is used by __repr__
        # for general cases self.filename is good
        # But for URL you could mask some part of the URL to keep
        # the main part.
        return self.dirname

    def _parse_header(self):
        nb_block = 0
        nb_segment = []
        sig_channels = []
        event_channels = []
        unit_channels = []
        self._asig_mmap = {}
        self._events_mmap = {}
        self._unit_mmap = {}

        for bl_index, bl_dir in enumerate(os.listdir(self.dirname)):
            if os.path.isdir(os.path.join(self.dirname, bl_dir)):
                nb_block += 1
                self._asig_mmap[bl_index] = {}
                self._events_mmap[bl_index] = {}
                self._unit_mmap[bl_index] = {}
                nb_segment.append(0)
                for seg_index, seg_dir in enumerate(os.listdir(os.path.join(
                                                               self.dirname,
                                                               bl_dir))):
                    nb_segment[bl_index] += 1
                    with open(os.path.join(self.dirname, bl_dir, seg_dir,
                        'structure.oebin')) as f:
                        seg_dict = json.load(f)

                    # continuous header
                    seg_sr = seg_dict['continuous'][0]['sample_rate']
                    self._sampling_rate = seg_sr
                    proc_dir = seg_dict['continuous'][0]['folder_name']
                    nchan = seg_dict['continuous'][0]['num_channels']
                    for chan_id, chan_dict in enumerate(
                        seg_dict['continuous'][0]['channels']):
                        dtype = 'int16'
                        units = chan_dict['units']
                        gain = chan_dict['bit_volts']
                        offset = 0.
                        chan_name = chan_dict['channel_name']
                        group_id = 0.
                        curr_chan = (chan_name, chan_id, seg_sr, dtype,
                                     units, gain, offset, group_id)
                        if curr_chan not in sig_channels:
                            sig_channels.append(curr_chan)

                    # continuous memmap
                    self._asig_mmap[bl_index][seg_index] = {}
                    timestamp_file = os.path.join(self.dirname, bl_dir, seg_dir,
                                                  'continuous', proc_dir,
                                                  'timestamps.npy')
                    self._asig_mmap[bl_index][seg_index]['timestamps'] = np.load(
                        timestamp_file, mmap_mode='r')
                    nsamps = self._asig_mmap[bl_index][seg_index]['timestamps'].shape[0]
                    data_file = os.path.join(
                        self.dirname, bl_dir, seg_dir, 'continuous', proc_dir,
                        'continuous.dat')
                    self._asig_mmap[bl_index][seg_index]['data'] = np.memmap(
                        data_file, dtype='int16', mode='r',
                        shape=(nchan, nsamps))

                    # events
                    self._events_mmap[bl_index][seg_index] = []
                    for chan_id, chan_dict in enumerate(seg_dict['events']):
                        # events header
                        proc_dir = chan_dict['folder_name']
                        name = chan_dict['channel_name']
                        event_type = 'event'
                        if (name, chan_id, event_type) not in event_channels:
                            event_channels.append((name, chan_id, event_type))

                        # events memmap
                        self._events_mmap[bl_index][seg_index].append({})
                        event_path = os.path.join(self.dirname, bl_dir, seg_dir,
                                                 'events', proc_dir)
                        self._events_mmap[bl_index][seg_index][chan_id]['timestamps'] = np.load(
                            os.path.join(event_path, 'timestamps.npy'), mmap_mode='r')
                        self._events_mmap[bl_index][seg_index][chan_id]['channels'] = np.load(
                            os.path.join(event_path, 'channels.npy'), mmap_mode='r')
                        if os.path.isfile(os.path.join(event_path, 'metadata.npy')):
                            self._events_mmap[bl_index][seg_index][chan_id]['metadata'] = np.load(
                                os.path.join(event_path, 'metadata.npy'), mmap_mode='r')
                        if os.path.isfile(os.path.join(event_path, 'text.npy')):
                            self._events_mmap[bl_index][seg_index][chan_id]['type'] = 'text'
                            self._events_mmap[bl_index][seg_index][chan_id]['text'] = np.load(
                                os.path.join(event_path, 'text.npy'), mmap_mode='r')
                        if os.path.isfile(os.path.join(event_path, 'data_array.npy')):
                            self._events_mmap[bl_index][seg_index][chan_id]['type'] = 'binary'
                            self._events_mmap[bl_index][seg_index][chan_id]['data_array'] = np.load(
                                os.path.join(event_path, 'data_array.npy'), mmap_mode='r')
                        if os.path.isfile(os.path.join(event_path, 'channel_states.npy')):
                            self._events_mmap[bl_index][seg_index][chan_id]['type'] = 'TTL'
                            self._events_mmap[bl_index][seg_index][chan_id]['channel_states'] = np.load(
                                os.path.join(event_path, 'channel_states.npy'), mmap_mode='r')
                        if os.path.isfile(os.path.join(event_path, 'full_words.npy')):
                            self._events_mmap[bl_index][seg_index][chan_id]['full_words'] = np.load(
                                os.path.join(event_path, 'full_words.npy'), mmap_mode='r')

                    # spikes
                    self._unit_mmap[bl_index][seg_index] = []
                    for spk_proc_id, grp_dict in enumerate(seg_dict['spikes']):
                        # spikes header
                        proc_dir = grp_dict['folder_name']
                        proc_name = grp_dict['source_processor']
                        nb_unit = grp_dict['num_channels']
                        wf_sr = grp_dict['sample_rate']
                        wf_left_sweep = grp_dict['pre_peak_samples']
                        wf_right_sweep = grp_dict['post_peak_samples']
                        wf_gain = 0.195  # not reported by format # TODO: read params from associated data processor
                        wf_offset = 0.
                        wf_units = 'uV'
                        self._unit_proc = {}
                        for unit_id, unit_dict in enumerate(grp_dict['channels']):
                            self._unit_proc[unit_id] = spk_proc_id
                            unit_name = unit_dict['channel_name']
                            curr_unit = (unit_name, unit_id, wf_units, wf_gain,
                                        wf_offset, wf_left_sweep, wf_sr)
                            if curr_unit not in unit_channels:
                                unit_channels.append(curr_unit)

                        # spikes memmap
                        self._unit_mmap[bl_index][seg_index].append({})
                        spikes_path = os.path.join(self.dirname, bl_dir, seg_dir,
                                                  'spikes', proc_dir)
                        self._unit_mmap[bl_index][seg_index][spk_proc_id]['spike_times'] = np.load(
                            os.path.join(spikes_path, 'spike_times.npy'), mmap_mode='r')
                        self._unit_mmap[bl_index][seg_index][spk_proc_id]['spike_electrode_id'] = np.load(
                            os.path.join(spikes_path, 'spike_electrode_indices.npy'), mmap_mode='r')
                        self._unit_mmap[bl_index][seg_index][spk_proc_id]['spike_cluster_id'] = np.load(
                            os.path.join(spikes_path, 'spike_clusters.npy'), mmap_mode='r')
                        self._unit_mmap[bl_index][seg_index][spk_proc_id]['spike_waveforms'] = np.load(
                            os.path.join(spikes_path, 'spike_waveforms.npy'), mmap_mode='r')
                        if os.path.isfile(os.path.join(spikes_path, 'metadata.npy')):
                            self._unit_mmap[bl_index][seg_index][spk_proc_id]['metadata'] = np.load(
                            os.path.join(spikes_path, 'metadata.npy'), mmap_mode='r')

        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)

        self.header = dict()
        self.header['nb_block'] = nb_block
        self.header['nb_segment'] = nb_segment
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels
        self._generate_minimal_annotations()

        self._t_starts = []
        self._t_stops = []
        for bl in range(self.header['nb_block']):
            self._t_starts.append([])
            self._t_stops.append([])
            for seg in range(self.header['nb_segment'][bl]):
                # import ipdb; ipdb.set_trace()
                self._t_starts[bl].append(self._asig_mmap[bl][seg]['timestamps'][0])
                self._t_stops[bl].append(self._asig_mmap[bl][seg]['timestamps'][-1])

    def _segment_t_start(self, block_index, seg_index):
        # this must return an float scale in second
        # this t_start will be shared by all object in the segment
        # except AnalogSignal
        return self._t_starts[block_index][seg_index] / self._sampling_rate

    def _segment_t_stop(self, block_index, seg_index):
        # this must return an float scale in second
        return self._t_stops[block_index][seg_index] / self._sampling_rate

    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        # this must return an int = the number of sample
        # Note that channel_indexes can be ignored for most cases
        # except for several sampling rate.
        return self._asig_mmap[block_index][seg_index]['data'].shape[1]

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        # This give the t_start of signals.
        # Very often this equal to _segment_t_start but not
        # always.
        # this must return an float scale in second

        # Note that channel_indexes can be ignored for most cases
        # except for several sampling rate.
        return self._segment_t_start(block_index, seg_index)

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        # this must return a signal chunk limited with
        # i_start/i_stop (can be None)
        # channel_indexes can be None (=all channel) or a list or numpy.array
        # This must return a numpy array 2D (even with one channel).
        # This must return the orignal dtype. No conversion here.
        # This must as fast as possible.
        # Everything that can be done in _parse_header() must not be here.
        # convertion to real units is done with self.header['signal_channels']
        if i_start is None:
            i_start = self._t_starts[block_index][seg_index]
        if i_stop is None:
            i_stop = self._t_stops[block_index][seg_index]
        if channel_indexes is None:
            channel_indexes = range(self._asig_mmap[block_index][seg_index]['data'].shape[0])
        return self._asig_mmap[block_index][seg_index]['data'][channel_indexes, i_start:i_stop]

    def _spike_count(self, block_index, seg_index, unit_index):
        # Must return the nb of spike for given (block_index, seg_index, unit_index)
        spk_proc_id = self._get_spk_proc_id(unit_index)
        return self._unit_mmap[block_index][seg_index][spk_proc_id]['spike_times']\
            [self._unit_mmap[block_index][seg_index][spk_proc_id]['spike_electrode_id']\
            ==unit_index].shape[0]

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        proc_id = self._get_spk_proc_id(unit_index)
        spike_ts = self._unit_mmap[block_index][seg_index][proc_id]['spike_times']\
            [self._unit_mmap[block_index][seg_index][proc_id]['spike_electrode_id']\
            ==unit_index]
        if t_start is not None:
            spike_ts = spike_ts[spike_ts >= int(t_start)]
        if t_stop is not None:
            spike_ts = spike_ts[spike_ts <= int(t_stop)]
        return spike_ts

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        # must rescale to second a particular spike_timestamps
        # with a fixed dtype so the user can choose the precisino he want.
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= self._sampling_rate  # because 10kHz
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        # this must return a 3D numpy array (nb_spike, nb_channel, nb_sample)
        # in the original dtype
        # this must be as fast as possible.
        # the same clip t_start/t_start must be used in _spike_timestamps()
        proc_id = self._get_spk_proc_id(unit_index)
        # which spikes to return
        spk_chan_bool = self._unit_mmap[block_index][seg_index][proc_id]['spike_electrode_id'] == unit_index
        if t_start is not None:
            spk_t_start_bool = spike_ts >= int(t_start)
        else:
            spk_t_start_bool = spk_chan_bool
        if t_stop is not None:
            spk_t_stop_bool = spike_ts[spike_ts <= int(t_stop)]
        else:
            spk_t_stop_bool = spk_chan_bool
        wf =  self._unit_mmap[block_index][seg_index][proc_id]['spike_waveforms'][spk_chan_bool & spk_t_start_bool & spk_t_stop_bool, :]
        return np.reshape(wf, (wf.shape[0], 1, wf.shape[1]))

    def _event_count(self, block_index, seg_index, event_channel_index):
        return self._events_mmap[block_index][seg_index][event_channel_index]['timestamps'].shape[0]

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        timestamps = self._events_mmap[block_index][seg_index][event_channel_index]['timestamps']
        if t_start is not None:
            timestamps = timestamps[timestamps>=t_start * self._sampling_rate]
        if t_stop is not None:
            timestamps = timestamps[timestamps<=t_stop * self._sampling_rate]
        duration = None
        # import ipdb; ipdb.set_trace()
        labels = ['{}'.format(self.header['event_channels'][event_channel_index][0]) for ts in timestamps]
        return timestamps, duration, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        return event_timestamps.astype(dtype) / self._sampling_rate

    def _rescale_epoch_duration(self, raw_duration, dtype):
        return None

    def _get_spk_proc_id(self, unit_index):
        return self._unit_proc[unit_index]

    def _get_event_proc_id(self, event_channel_index):
        return self._unit_proc[unit_index]
