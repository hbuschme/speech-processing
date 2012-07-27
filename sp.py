#!/usr/bin/env python
# -*- coding: utf-8 -*-

# speech-processing -- A Python framework for speech processing
# Copyright (C) 2010 Hendrik Buschmeier
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division, print_function

import collections
import csv
import math
import operator
import Queue
import random
import sys
import threading
import time

import numpy
import pyaudio
import scikits.audiolab
import scipy.ndimage


## --- Miscellaneous Functions and Datastructures ----------------------------

def start_thread(target, deamonic=True):
    """Conveniently start a function in a new thread."""
    t = threading.Thread(target=target)
    t.daemon = deamonic
    t.start()


class LTRQueue(Queue.Queue):
    """A left-to-right deque-Queue mixture.
    
    This class combines the blocking functionality of Queue.Queue with
    the maxlen functionality of collection.deque in a left-to-right 
    fashion. Elements are appended left and popped right.
    """
    
    def __init__(self, maxsize=0, maxlen=0):
        # Unfortunately, Queue is an "old-style class" and thus we need to
        # call its constructor in the following way:
        Queue.Queue.__init__(self, maxsize)
        self._init(maxlen)
    
    def _init(self, maxsize):
        self.queue = collections.deque(maxlen=maxsize)
    
    def _qsize(self, len=len):
        return len(self.queue)
    
    def _put(self, item):
        self.queue.appendleft(item)
    
    def _get(self):
        return self.queue.pop()


class Utterance(object):
    
    def __init__(self, audio_frames=[]):
        super(Utterance, self).__init__()
        self._audio_frames = collections.deque(audio_frames)
        self.properties = {}
    
    def add_frame(self, frame):
        self._audio_frames.appendleft(frame)
    
    def pitch_values(self):
        values = []
        for frame in self._audio_frames:
            values += frame['pitch']
        return values
    
    def duration(self):
        first_frame = self._audio_frames[-1]
        last_frame = self._audio_frames[0]
        last_frame_len = 1 / last_frame['samplerate'] * last_frame['nr-samples']
        return last_frame['timestamp'] + last_frame_len - first_frame['timestamp']
    
    def speech_duration(self):
        first_speech_frame = None
        last_speech_frame = None
        for frame in self._audio_frames:
            if not frame['silence']:
                first_speech_frame = frame
        for frame in reversed(self._audio_frames):
            if not frame['silence']:
                last_speech_frame = frame
        if first_speech_frame is not None and last_speech_frame is not None:
            last_speech_frame_len = 1 / last_speech_frame['samplerate'] * last_speech_frame['nr-samples']
            return last_speech_frame['timestamp'] + last_speech_frame_len - first_speech_frame['timestamp']
        return 0     


## --- Processor Interfaces --------------------------------------------------

class DataConsumer(object):
    
    def __init__(self, buffer_size=5):
        super(DataConsumer, self).__init__()
        self._buffer_size = buffer_size
        self._data_queue = LTRQueue(maxlen=buffer_size)
    
    def push(self, data):
        """Push data into this consumers queue."""
        if self._data_queue.qsize() == self._buffer_size:
            print(self.__class__.__name__, " will probably lose a frame.", file=sys.stderr)
        self._data_queue.put_nowait(data)
        self.handle_push_event()
    
    def handle_push_event(self):
        """Event handler that is called when data is pushed into queue."""
        pass
        
    def has_data(self):
        """Return whether there is data in the queue."""
        return not self._data_queue.empty()
    
    def pop(self, block=True, timeout=None):
        """Return and remove oldest element in the queue; block when
        queue is empty.
        """
        return self._data_queue.get(block, timeout)
    
    def pop_nowait(self):
        """Return and remove oldest element in queue or None if queue
        is empty.
        """
        return self._data_queue.get(block=False)


class DataProducer(object):
    
    def __init__(self):
        super(DataProducer, self).__init__()
        self._consumers = []
    
    def register_data_consumers(self, consumers):
        """Register one or more data consumers."""
         # Is it a list of DataConsumers ...
        if isinstance(consumers, collections.Iterable):
            for consumer in consumers:
                if isinstance(consumer, DataConsumer):    
                    self._consumers.append(consumer)
                else:
                    raise TypeError("Object is not of type DataConsumer.")
         # ... or it just a single DataConsumer object?
        elif isinstance(consumers, DataConsumer):
            self._consumers.append(consumers)
        else:
            raise TypeError("Object is not of type DataConsumer.")
    
    def push_to_data_consumers(self, data):
        """Push data to registered data consumers."""
        for consumer in self._consumers:
            consumer.push(data)


## --- Audio I/O -------------------------------------------------------------

class AudioRecorder(DataProducer):
    """A data producer that reads audio data from the choses input
    device.
    """
    
    def __init__(self, sample_rate=44100, f0_min=75, input_device_index=0): 
        super(AudioRecorder, self).__init__()
        self._portaudio = pyaudio.PyAudio()
        self._format = pyaudio.paInt16
        self._sample_rate = sample_rate
        self._f0_min = f0_min
        self._input_device_index = input_device_index
        self._frame_length = int(self._sample_rate // self._f0_min)
    
    def __del__(self):
        """Close portaudio."""
        self._portaudio.terminate()
    
    def get_audio_samples(self, buffer_size):
        """Returns an audio samples from the microphone. This is a
        generator function.
        """
        audio_stream = self._portaudio.open(
            format = self._format,
            channels = 1,
            rate = self._sample_rate,
            input = True,
            input_device_index = self._input_device_index,
            frames_per_buffer = buffer_size)
        while self._enabled:
            try:
                t = time.time()
                frame = numpy.fromstring(audio_stream.read(buffer_size), dtype=numpy.int16)
                yield (t, numpy.true_divide(frame, numpy.array([32767])))
            except(IOError):
                print("An IOError occured during audio recording, lost 1 frame.", file=sys.stderr)
        audio_stream.stop_stream()
        audio_stream.close()
    
    def run(self):
        self._enabled = True
        ident = 0
        for t, audio_frame in self.get_audio_samples(self._frame_length):
            self.push_to_data_consumers({
                'id' : ident,
                'timestamp' : t,
                'samplerate' : self._sample_rate,
                'nr-samples' : len(audio_frame),
                'audio' : audio_frame
            })
            ident += 1


class FileReader(DataProducer):
    """A data producer that reads audio data from the specified file."""
    
    def __init__(self, filename):
        super(FileReader, self).__init__()
        self.filename = filename
        
    def open_file(self):
        self.file = scikits.audiolab.Sndfile(self.filename, 'r')
        
    def read(self, buffer_size=640):
        while True:
            try:
                data = self.file.read_frames(buffer_size)
                self.push_to_data_consumers({'audio':data})
                time.sleep(0.1)
            except RuntimeError as e:
                print(e)
                break
        
    def run(self):
        self.open_file()
        self.read()


class FileWriter(DataConsumer):
    """A data consumer that writes the received audio frames into a wav
     file.
     """
    
    def __init__(self, sample_rate=44100):
        super(FileWriter, self).__init__(100) # Needs a longer buffer
        self._sample_rate = sample_rate
        self.data = numpy.array([], dtype=numpy.float32)
        self.file = None
    
    def create_new_file(self, filename):
        print("Created file.")
        format = scikits.audiolab.Format('wav')
        self.file = scikits.audiolab.Sndfile(filename, 'w', format, 1, self._sample_rate)
    
    def run(self, length=5):
        i = 1
        while True:
            frame = self.pop()
            if frame['vad'] == 'START':
                self.create_new_file(str(i) + '.wav')
                i += 1
            if self.file is not None:
                self.file.write_frames(frame['audio'])
                self.file.sync()
            if frame['vad'] == 'END':
                if self.file is not None:
                    self.file.close()
                    print("Wrote file.")


class UtteranceWriter(DataConsumer):
    
    def __init__(self, sample_rate=44100):
        super(UtteranceWriter, self).__init__()
        self._sample_rate = sample_rate
    
    def create_new_file(self, filename):
        format = scikits.audiolab.Format('wav')
        self.file = scikits.audiolab.Sndfile(filename, 'w', format, 1, self._sample_rate)
        self.data = numpy.array([], dtype=numpy.float32)
    
    def run(self):
        id = 0
        while True:
            utterance = self.pop()
            self.create_new_file('test/' + str(id) + '.wav')
            n = 0
            for frame in reversed(utterance._audio_frames):
                n+=1
                self.file.write_frames(frame['audio'])
                print(frame['vad'], end=" ")
                self.file.sync()
            self.file.close()
            print("Wrote file " + str(id) + ', ' + str(n)  + 'frames.')
            id +=1


## --- Audio Frame Parameter Estimation --------------------------------------

class ParameterEstimator(DataConsumer, DataProducer):
    """A data consumer and producer that enriches audio frames with
    energy, root mean square and zero crossing rate information."""
    
    def __init__(self, rms=True, zcr=True, energy=True):
        super(ParameterEstimator, self).__init__()
        self.compute_rms = rms
        self.compute_zcr = zcr
        self.compute_energy = energy
    
    def energy(self, frame):
        """Calculate the energy of a frame. 
        Roughly implemented after Rabiner, L. R. and Sambur M. R. 
        (1975). An Algorithm for Determining the Endpoints of Isolated
        Utterances. The Bell Systems Technical Journal, 54:297--315.
        """
        return numpy.sum(numpy.abs(frame))
    
    def rms(self, frame):
        """Calculate the root mean square of a frame."""
        return math.sqrt(numpy.mean(numpy.square(frame)))
    
    def zcr(self, frame):
        """Calculate the zero crossing rate of a frame.
        Implemented after Chen, C.H. (1988). Signal Processing Handbook.
        p. 531, New York: Dekker.
        """
        T = len(frame) - 1
        return 1 / T * numpy.sum(numpy.signbit(numpy.multiply(frame[1:T], frame[0:T - 1])))
    
    def run(self):
        while True:
            frame = self.pop()
            if self.compute_rms:
                frame['rms'] = self.rms(frame.get('audio'))
            if self.compute_zcr: 
                frame['zcr'] = self.zcr(frame.get('audio'))
            if self.compute_energy:
                frame['energy'] = self.energy(frame.get('audio'))
            self.push_to_data_consumers(frame)


## --- Voice Activity Detection ----------------------------------------------

class VoiceActivityDetectionState(object):
    """Implements an abstract state of a state machine according to the
    "state" design pattern.
    """
    
    def __init__(self, vad, silence_queue, speech_queue):
        super(VoiceActivityDetectionState, self).__init__()
        self.vad = vad
        self.silence_queue = silence_queue
        self.speech_queue = speech_queue
    
    def enter(self):
        """Abstract method called when the state is entered."""
        pass
        
    def exit(self):
        """Abstract method called when the state is left."""
        pass
    
    def handle_silence_frame(self, frame):
        """Abstract method to handle a silence frame."""
        pass
    
    def handle_speech_frame(self, frame):
        """Abstract method to handle a speech frame."""
        pass
    
    def __str__(self):
        """Return the name of this state."""
        return self.__class__.__name__


class SilenceState(VoiceActivityDetectionState):
    
    def handle_silence_frame(self, frame):
        self.silence_queue.appendleft(frame)
    
    def handle_speech_frame(self, frame):
        if len(self.silence_queue) == 0:
            self.silence_queue.appendleft(frame)
        else:
            self.speech_queue.appendleft(frame)
        self.vad.change_state(self.vad.ONSETTING)


class OnsettingState(VoiceActivityDetectionState):
    
    def handle_silence_frame(self, frame):
        self.silence_queue.clear()
        self.speech_queue.clear()
        self.silence_queue.appendleft(frame)
        self.vad.change_state(self.vad.SILENCE)
    
    def handle_speech_frame(self, frame):
        if len(self.speech_queue) < self.vad.onset_threshold:
            self.speech_queue.appendleft(frame)
        else:
            self.speech_queue.appendleft(frame)
            ff_marked = False
            while len(self.silence_queue) > 0:
                _frame = self.silence_queue.pop()
                _frame['vad'] = 'PRE-SPEECH' if ff_marked else 'START'
                ff_marked = True
                self.vad.push_to_data_consumers(_frame)
                time.sleep(0.001)
            while len(self.speech_queue) > 0:
                _frame = self.speech_queue.pop()
                _frame['vad'] = 'SPEECH'
                self.vad.push_to_data_consumers(_frame)
                time.sleep(0.001)
            self.vad.change_state(self.vad.SPEAKING)


class SpeakingState(VoiceActivityDetectionState):
    
    def handle_silence_frame(self, frame):
        self.speech_queue.appendleft(frame)
        self.vad.change_state(self.vad.ENDING)
    
    def handle_speech_frame(self, frame):
        frame['vad'] = "SPEECH"
        self.vad.push_to_data_consumers(frame)


class EndingState(VoiceActivityDetectionState):
    
    def handle_silence_frame(self, frame):
        if len(self.speech_queue) < self.vad.ending_threshold:
            self.speech_queue.appendleft(frame)
        else:
            self.speech_queue.appendleft(frame)
            ctx = self.vad.context_width
            while len(self.speech_queue) > 0 and ctx > 0:
                _frame = self.speech_queue.pop()
                _frame['vad'] = 'POST-SPEECH' if ctx > 1 else 'END'
                self.vad.push_to_data_consumers(_frame)
                time.sleep(0.001)
                ctx -= 1
            self.silence_queue.clear()                
            self.speech_queue.clear()
            self.vad.change_state(self.vad.SILENCE)
    
    def handle_speech_frame(self, frame):
        while len(self.speech_queue) > 0:
            _frame = self.speech_queue.pop()
            _frame['vad'] = 'PAUSE'
            self.vad.push_to_data_consumers(_frame)
        frame['vad'] = 'SPEECH'
        self.vad.push_to_data_consumers(frame)
        self.vad.change_state(self.vad.SPEAKING)


class VoiceActivityDetector(DataConsumer, DataProducer):
    """A simple voice activity detection processor. Discards silence
    audio frames and enriches non-silent audio frames with voice
    activity information.
    """
    
    def __init__(self, silence_threshold=0.005, context_width=5, onset_threshold=5, ending_threshold=20):
        super(VoiceActivityDetector, self).__init__()
        self.silence_threshold=silence_threshold
        self.context_width = context_width
        self.onset_threshold = onset_threshold
        self.ending_threshold = ending_threshold
        self._silence_q = collections.deque(maxlen=context_width)
        self._speech_q = collections.deque()
        self.SILENCE = SilenceState(self, self._silence_q, self._speech_q)
        self.ONSETTING = OnsettingState(self, self._silence_q, self._speech_q)
        self.SPEAKING = SpeakingState(self, self._silence_q, self._speech_q)
        self.ENDING = EndingState(self, self._silence_q, self._speech_q)
        self._state = self.SILENCE
    
    def handle_frame(self, frame):
        """Dispatches the frame to the relevant method of the current
        state object.
        """
        if frame['silence'] == True:
            self._state.handle_silence_frame(frame)
        else:
            self._state.handle_speech_frame(frame)
    
    def change_state(self, state):
        """Changes the state of this state machine."""
        self._state.exit()
        self._state = state
        self._state.enter()
    
    def evaluate(self, frame):
        """Enriches the given audio frame with silence and voicing
        information.
        """
        frame['silence'] = True if frame['rms'] < self.silence_threshold else False
        frame['voiced']  = True if frame['zcr'] < 0.1 else False
    
    def run(self):
        while True:
            frame = self.pop()
            self.evaluate(frame)
            self.handle_frame(frame)


## --- Pitch Tracking --------------------------------------------------------

class YinPitchEstimator(DataConsumer):
    """Implementation of the YIN fundamental frequency estimation 
    algorithm, steps 2, 3 and 4 as described in:
    de Cheveigne, A. and Kawahara, H. (2002). YIN, a fundamental 
    frequency estimator for speech and music. Journal of the Acoustical
    Society of America, 111:1917--1930.
    """
    
    def __init__(self, sample_rate=44100, f0_min=75, f0_max=500, use_optimized_version=False):
        """Initialise YIN, set sample rate and min/max pitch values."""
        super(YinPitchEstimator, self).__init__(buffer_size=None)
        self._sample_rate = sample_rate
        self._f0_min = f0_min
        self._f0_max = f0_max
         # Warning: optimized version seems to gain false results
        self._use_optimized_version = use_optimized_version
        self._delay_range = range(1, int(self._sample_rate // self._f0_min))
        self._compute_sparse_stuff()
        self._min_frame_size = self._delay_range[-1]
    
    def _compute_sparse_stuff(self):
        """Compute some sparse data structures, which are needed when
        using the optimized version of this algorithm.
        """
        d = {1:(1,1,1)}
        for f in range(int(self._f0_min), int(self._sample_rate / 2)):
            bin_mean = int(round(self._sample_rate / f))
            if not bin_mean in d:
                bin_min  = int(round(self._sample_rate / (f + 0.999999999)))
                bin_max  = int(round(self._sample_rate / (f + 0.0)))
                d[bin_mean] = (bin_mean, bin_min, bin_max)
        self._sparse_delay_range = sorted(d.keys())
        self._sparse_delay_range_mapping = []
        self._pseudo_sparse_delay_range = []
        x = sorted(d.values(), key=operator.itemgetter(0))
        i = 0
        for delay in self._delay_range:
            self._sparse_delay_range_mapping.append((delay, i))
            self._pseudo_sparse_delay_range.append(self._sparse_delay_range[i])
            if delay == x[i][2]:
                i += 1
        
    def _difference(self, delay, frame):
        """Compute the difference between a frame and itself shifted to
        the right by delay samples (step 2, eq. 6).
        """
        ds = numpy.subtract(frame[0:-delay], frame[delay:])
        return numpy.sum(numpy.multiply(ds, ds))
    
    def _difference_with_cache(self, delay, frame, cache):
        """Compute the difference between a frame and itself shifted to
        the right by delay samples (step 2, eq. 6), This is only done
        if this delay has not yet been computed (use only when fed with
        pseudo sparse delay range).
        """
        if not delay in cache:
            ds = numpy.subtract(frame[0:-delay], frame[delay:])
            cache[delay] = numpy.sum(numpy.multiply(ds, ds))
        return cache[delay]
    
    def _compute_differences(self, frame):
        """Compute the differences for all relevant delay values (step 2)."""
        return numpy.array([self._difference(delay, frame) for delay in self._delay_range])
    
    def _compute_differences_pseudo_sparsely(self, frame):
        """Compute the differences for all relevant delay values
        (step 2), tries to save some steps of the computation."""
        cache = {}
        return numpy.array([self._difference_with_cache(delay, frame, cache) for delay in self._pseudo_sparse_delay_range])
    
    def _cumulative_mean_normalized_differences(self, frame):
        """Compute the cumulative mean normalised difference (step 3)."""
        cached_differences = self._compute_differences_pseudo_sparsely(frame) if self._use_optimized_version else self._compute_differences(frame)
        return [cached_differences[delay - 1] / ((1 / delay) * numpy.sum(cached_differences[0:delay])) for delay in self._delay_range]
    
    def _find_minima(self, cmnds, search_window_size=10):
        """Find minima in cumulative mean normalised difference (step 4)."""
        x = (cmnds == scipy.ndimage.minimum_filter(cmnds, search_window_size))
        return numpy.nonzero(x)[0]
    
    def _choose_min_delay(self, cmnds, minima, abs_threshold=0.1):
        """Choose the minmum with the smallest delay value (step 4)."""
        below = filter(lambda y: 0 <= cmnds[y] <= abs_threshold, minima)
        return below[0] if len(below) > 0 else 0
    
    def _delay_to_frequency(self, delay):
        """Convert delay (in samples) to frequency rate."""
        return self._sample_rate / delay
    
    def get_pitch(self, frame):
        """Get the pitch value of a frame."""
        if len(frame) < self._min_frame_size:
            raise Exception('Frame contains fewer samples than needed ({0} < {1})'.format(len(frame), self._min_frame_size))
            return 0
        cmnds = self._cumulative_mean_normalized_differences(frame)
        min_delay = (self._choose_min_delay(cmnds, self._find_minima(cmnds)))
        pitch = self._delay_to_frequency(min_delay) if min_delay != 0 else 0
        return pitch if pitch <= self._f0_max else 0
        #Todo: choose freq based on beam width and last pitch
    
    def run(self):
        while True:
            frame = self.pop()
            before = time.time()
            pitch = self.get_pitch(frame['audio'])
            #print(pitch, "Yin Processing time: ", time.time() - before, "frame_delay", time.time() - frame['timestamp'], "Qsize", self._data_queue.qsize())
    


class PitchTracker(DataConsumer, DataProducer):
    
    def __init__(self, sample_rate=44100, f0_min=75, no_partitions=2):
        super(PitchTracker, self).__init__(buffer_size=None)
        self._sample_rate = sample_rate
        self._f0_min = f0_min
        self._no_partitions = no_partitions
        self._frame_length = int(self._sample_rate // self._f0_min)
        self.processing_queue = collections.deque()
        self.yin = YinPitchEstimator(sample_rate = self._sample_rate, f0_min=self._f0_min)
        self._slicing_points = self._precompute_slicing_points()
    
    def _precompute_slicing_points(self):
        r = self._frame_length / self._no_partitions
        return [(int(round(i*r)), int(round(i*r+r-1))) for i in range(0, self._no_partitions)]
        
    def process(self):
        if len(self.processing_queue) == 1:
            # The processing queue contains only one frame, i.e., a starting
            # frame with no right context. This frame only gets a single pitch
            # value and remains in the queue.
            frame = self.processing_queue[-1]
            frame['nr-pitch-values'] = 1
            frame['pitch'] = [self.yin.get_pitch(frame['audio'])]
        elif len(self.processing_queue) > 1:
            # The processing queue contains more than one frame. The rightmost
            # frame is popped from the queue and used as right context for the
            # then rightmost frame. A sliding window is shifted N times over 
            # both frames and one pitch value is calculated for each window.
            context_frame = self.processing_queue.pop()
            frame = self.processing_queue[-1]
            frame['nr-pitch-values'] = self._no_partitions
            pitch_values = []
            for i in range(1, self._no_partitions + 1):
                if i < self._no_partitions:
                    # The window of each of the first N-1 partitions spans
                    # part of the context and part of the current frame.
                    pitch_values.append(self.yin.get_pitch(numpy.concatenate((context_frame['audio'][self._slicing_points[i][0]:], frame['audio'][:self._slicing_points[i - 1][1]]))))
                elif i == self._no_partitions:
                    # The window of the last partition spans the whole current
                    # frame
                    pitch_values.append(self.yin.get_pitch(frame['audio']))
            frame['pitch'] = pitch_values
            # The context frame is pushed to the consumers with a one frame
            # delay, if current frame is an END frame, push it to the data_
            # consumers, too.
            self.push_to_data_consumers(context_frame)
            if frame['vad'] == 'END':
                self.push_to_data_consumers(frame)
    
    def run(self):
        while True:
            self.processing_queue.appendleft(self.pop())
            self.process()


class VsuClassifier(DataConsumer, DataProducer):
    """Classifier for Very Short Utterances.

    Edlund, J., Heldner, M., & Pelcé, A. (2009). Prosodic features of
    very short utterances in dialogue. In Vainio, M., Aulanko, R., &
    Aaltonen, O. (Eds.), Nordic Prosody -- Proceedings of the Xth
    Conference, pp. 57--68. Frankfurt am Main: Peter Lang."""
    
    def __init__(self):
        super(VsuClassifier, self).__init__(5)
        self._vsu_candidate_utt = Utterance()
        self._duration_threshold = 2.0
    
    def clear_candidate(self):
        self._vsu_candidate_utt = Utterance()
    
    def classify(self):
        is_vsu = False
        if self._vsu_candidate_utt.duration() < self._duration_threshold:
            is_vsu = True
        self._vsu_candidate_utt.properties['vsu'] = is_vsu
    
    def run(self):
        while True:
            frame = self.pop()
            self._vsu_candidate_utt.add_frame(frame)
            if frame['vad'] == 'END':
                self.classify()
                self.push_to_data_consumers(self._vsu_candidate_utt)
                self.clear_candidate()


## --- Inspecting & Testing --------------------------------------------------

class FrameInspector(DataConsumer):
    
    def handle_push_event(self):
        if self.has_data():
            frame = self.pop()
            for value in frame.get('pitch', []):
                print(value, end=' ')
            #print(frame['vad'])
            print()


class UtteranceInspector(DataConsumer):
    
    def handle_push_event(self):
        if self.has_data():
            utt = self.pop()
            try:
                print("Duration: ", utt.duration())
                print("SpeechDuration: ", utt.speech_duration())
                print("#Frames:", len(utt._audio_frames))
                print("VSU?", utt.properties['vsu'])
                print("PITCH", len(utt.pitch_values()))
                print("SIMPLIFIED PITCH", utt.properties['simplified-pitch'])
            except: pass


## --- Main ------------------------------------------------------------------
if __name__ == '__main__':
    #sys.exit(1)
    SAMPLE_RATE = 11025
    # 0 = build-in, 2 = H2, (3 = H2 on linux)
    ar = AudioRecorder(sample_rate=SAMPLE_RATE, input_device_index=2)
    start_thread(ar.run)

    #fr = FileReader('audio-samples/vp22-inGF-01-slide01.wav')
    #start_thread(fr.run)
    
    pe = ParameterEstimator()
    ar.register_data_consumers(pe)
    #fr.register_data_consumers(pe)
    start_thread(pe.run)
    
    vad = VoiceActivityDetector()
    pe.register_data_consumers(vad)
    start_thread(vad.run)
    
    #fw = FileWriter(sample_rate=SAMPLE_RATE)
    #vad.register_data_consumers(fw)
    #fw.run()
    
    #ype = YinPitchEstimator(sample_rate=SAMPLE_RATE)
    #vad.register_data_consumers(ype)
    #ype.run()
    
    #pt = PitchTracker(no_partitions=1, sample_rate=SAMPLE_RATE)
    #vad.register_data_consumers(pt)
    #pt.register_data_consumers(FrameInspector())
    #start_thread(pt.run)
    #pt.run()
    
    #vc = VsuClassifier()
    #pt.register_data_consumers(vc)
    #vc.register_data_consumers(UtteranceInspector())
    #start_thread(vc.run)
    #vc.run()
    
    #uw = UtteranceWriter(sample_rate=SAMPLE_RATE)
    #vc.register_data_consumers(uw)
    #uw.run()