import logging
import nidaqmx
from nidaqmx.constants import (Level, 
                               Edge, 
                               Slope, 
                               TaskMode, 
                               TerminalConfiguration, 
                               AOIdleOutputBehavior, 
                               AcquisitionType as AcqType, 
                               FrequencyUnits as Freq,
                               SourceSelection
                               )
from time import sleep
from typing import Union

class NI:

    def __init__(self, dev_name: str, samples_per_sec: float, livestream_frequency_hz : int,
                 period_time_s: float, ao_channels: dict):
        """init.

        :param dev_name: NI device name as it appears in Device Manager.
        :param samples_per_sec: sample playback rate in samples per second.
        :param period_time_s: the total waveform period for one frame pattern.
        :param ao_channels: dict in the form of
            {<analog output name>: <analog output channel>}.
        """
        self.log = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.dev_name = dev_name
        self.dev = nidaqmx.system.device.Device(self.dev_name)
        self.log.warning('Resetting NIDAQ')
        self.dev.reset_device()
        self.samples_per_sec = samples_per_sec
        self.livestream_frequency_hz = livestream_frequency_hz
        self.period_time_s = period_time_s
        self.ao_names_to_channels = ao_channels
        # Total samples is the sum of the samples for every used laser channel.
        self.daq_samples = round(self.samples_per_sec * self.period_time_s)
        self.co_task = None
        self.ao_task = None
        self.live = None

    def configure(self, live: bool = False):
        """Configure the NI card to play either `frame_count` frames or
        continuously.

        :param frame_count: the number of frames to play waveforms for. If
            left unspecified, `live` must be true.
        :param live: if True, play the waveforms indefinitely. `frame_count`
            must be left unspecified in this case. Otherwise, play the
            waveforms for the specified `frame_count`.
        """
        self.live = live
        frequency = self.samples_per_sec/self.daq_samples if not live else self.livestream_frequency_hz
        self.close()    # If tasks exist then close them
        self.dev.reset_device()

        self.co_task = nidaqmx.Task('counter_output_task')
        co_chan = self.co_task.co_channels.add_co_pulse_chan_freq(
            self.get_channel_physical_name(channel_nb="0",type="ctr"), # counter 0
            units=Freq.HZ,
            idle_state=Level.LOW,
            initial_delay=0.0,
            freq=frequency,
            duty_cycle=0.5)
        co_chan.co_pulse_term = self.get_channel_physical_name(channel_nb="0",type="PFI")

        if live:
            self.set_pulse_count(pulse_count=0)

        self.ao_task = nidaqmx.Task("analog_output_task")
        for channel_name, channel_index in self.ao_names_to_channels.items():
            physical_name = self.get_channel_physical_name(channel_index, type = "ao")
            self.log.debug(f'Adding channel {physical_name}-{channel_name} to NI task')
            if "galvo" in channel_name.lower(): 
                # safeguard for negative/positive values to AO channels, used for galvos primarily
                # TODO :hard coded, need to make this config dependant
                min_val = -1
                max_val = 10
            elif any([laser in channel_name for laser in ["488","561","638","405"]]): 
                min_val = 0
                max_val = 10
            elif "etl" in channel_name.lower(): 
                min_val = 0
                max_val = 10
            else :
                min_val = -0.1 # safeguard for 0 to 5V values to AO channels
                max_val = 5.1
            self.log.debug(f'Channel {physical_name} voltage boundary values are {min_val} to {max_val}')
            channel = self.ao_task.ao_channels.add_ao_voltage_chan(physical_name, min_val=min_val, max_val=max_val)
            channel.ao_term_cfg = TerminalConfiguration.RSE # change the default AO chanel electrical termination configuration
            channel.ao_dac_ref_src = SourceSelection.INTERNAL
            # channel.ao_dac_ref_allow_conn_to_gnd = True
            try : 
                channel.ao_idle_output_behavior = AOIdleOutputBehavior.HIGH_IMPEDANCE
            except Exception as e:
                self.log.info("Could not set AOIdleOutputBehavior to HIGH_IMPEDANCE "
                              f"on channel {physical_name} for {channel_name}.")
                              # f"\nError : {e}")

        self.ao_task.timing.cfg_samp_clk_timing(
            rate=self.samples_per_sec,
            active_edge=Edge.RISING,
            sample_mode=AcqType.FINITE,
            samps_per_chan=self.daq_samples)
        self.ao_task.triggers.start_trigger.retriggerable = True
        # TODO: trigger source should be in the config.
        self.ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
#			trigger_source=f'/{self.dev_name}/PFI1',
            trigger_source= self.get_channel_physical_name(channel_nb="0",type="PFI"),
            trigger_edge=Slope.RISING)

        self.ao_task.out_stream.output_buf_size = self.daq_samples  # Sets buffer to length of voltages
        self.ao_task.control(TaskMode.TASK_COMMIT)

    def get_channel_physical_name(self, channel_nb : Union[int, str], type = "ao") -> str :
        physical_name = f"/{self.dev_name}/{type}{channel_nb}"
        return physical_name
    
    def assign_waveforms(self, voltages_t, scout_mode: bool = False):

        #if scout_mode:
        self.ao_task.control(TaskMode.TASK_UNRESERVE)   # Unreserve buffer
        self.ao_task.timing.cfg_samp_clk_timing(
            rate=self.samples_per_sec,
            active_edge=Edge.RISING,
            sample_mode=AcqType.FINITE,
            samps_per_chan=len(voltages_t[0]))
        self.ao_task.out_stream.output_buf_size = len(voltages_t[0])  # Sets buffer to length of voltages
        self.ao_task.control(TaskMode.TASK_COMMIT)

        new_v = []
        for (channel_name, channel_id), data in zip(self.ao_names_to_channels.items(), voltages_t) :
            self.log.debug(f"Writing channel ao{channel_id}-{channel_name} with :\n"
                           f"{len(data)} samples, "
                           f"max={max(data)}, min={min(data)}, "
                           f"element dtype ={type(data[0])}, sequence dtype = {type(data)}")
            new_v.append([float(item) for item in data])
        
        self.ao_task.write(voltages_t, auto_start=False)

    def set_pulse_count(self, pulse_count: int = None):
        """Set the number of pulses to generate or None if pulsing continuously.

        :param pulse_count: The number of pulses to generate. If 0 or
            unspecified, the counter pulses continuously.
        :return:
        """
        self.log.debug(f"Setting counter task count to {pulse_count} pulses.")
        optional_kwds = {}
        # Don't specify samps_per_chan to use default value if it was specified
        # as 0 or None.
        if pulse_count:
            optional_kwds['samps_per_chan'] = pulse_count
        self.co_task.timing.cfg_implicit_timing(
            sample_mode=AcqType.FINITE if pulse_count else AcqType.CONTINUOUS,
            **optional_kwds)

    def wait_until_done(self, timeout=1.0):
        return self.co_task.wait_until_done(timeout)


    def start(self):
        if self.ao_task:
            self.ao_task.start()
        # time.sleep(2*(self.daq_samples/self.update_freq)) # wait for last AO to play
        # print(2*(self.daq_samples/self.update_freq))
        if self.co_task:
            self.co_task.start()

    def stop(self, wait: bool = False, sleep_time = None):
        """Stop the tasks. Optional: try waiting first before stopping."""
        try:
            if wait:
                self.wait_until_done()
        finally:
            if self.co_task:
                self.co_task.stop()
            if sleep_time is not None:
                sleep(sleep_time)  # Sleep so ao task can finish
            # time.sleep(2*(self.daq_samples/self.update_freq)) # wait for last AO to play
            # print(2*(self.daq_samples/self.update_freq))
            if self.ao_task:
                self.ao_task.stop()

    def close(self):
        if self.co_task:
            self.co_task.close()
        if self.ao_task:
            self.ao_task.close()

