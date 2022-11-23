"""Abstraction of the ExaSPIM Instrument."""

import numpy as np
from math import ceil
from pathlib import Path
from psutil import virtual_memory
from time import perf_counter, sleep
from tifffile import TiffWriter
from mock import NonCallableMock as Mock
from exaspim.exaspim_config import ExaspimConfig
from exaspim.devices.camera import Camera
from exaspim.devices.waveform_generator import NI
from exaspim.processes.mip_processor import MIPProcessor
from exaspim.processes.stack_writer import StackWriter
from exaspim.processes.file_transfer import FileTransfer
from exaspim.processes.data_logger import DataLogger
from exaspim.data_structures.shared_double_buffer import SharedDoubleBuffer
from tigerasi.tiger_controller import TigerController, UM_TO_STEPS
from tigerasi.sim_tiger_controller import TigerController as SimTiger
# TODO: consolodate these later.
from mesospim.spim_base import Spim
from mesospim.devices.tiger_components import SamplePose


class Exaspim(Spim):

    def __init__(self, config_filepath: str, simulated: bool = False):

        super().__init__(config_filepath, simulated)
        self.cfg = ExaspimConfig(config_filepath)

        # Separate Processes. Note: most of these are per-channel.
        self.data_logger_worker = None  # External Memento data logger.
        self.mip_workers = {}  # aggregates MIPs from frames

        # Hardware
        self.cam = Camera() if not self.simulated else Mock(Camera)
        self.ni = NI() if not self.simulated else Mock(NI)
        self.etl = None
        self.gavlo_a = None
        self.gavlo_b = None
        self.daq = None
        self.tigerbox = TigerController(**self.cfg.tiger_obj_kwds) if not \
            self.simulated else SimTiger(**self.cfg.tiger_obj_kwds)
        self.sample_pose = SamplePose(self.tigerbox,
                                      **self.cfg.sample_pose_kwds)

        # Extra Internal State attributes for the current image capture
        # sequence. These really only need to persist for logging purposes.
        self.image_index = 0  # current image to capture.
        self.total_tiles = 0  # tiles to be captured.
        self.stage_x_pos = None
        self.stage_y_pos = None

        # TODO: consider SharedMemory double buffer for working with MIPS
        #   and Imaris Writer data compressor in another process.
        # TODO: do the minimum (ideally zero) number of image transformations
        #   before handing them off to Imaris Writer.

        # Setup hardware according to the config.
        self._setup_motion_stage()
        self._setup_camera()

    def _setup_motion_stage(self):
        """Configure the sample stage according to the config."""
        # The tigerbox x axis is the sample pose z axis.
        #   TODO: map this in the config.
        self.tigerbox.set_axis_backlash(x=0)

    def _setup_camera(self):
        """Configure the camera according to the config."""
        # pass config parameters into object here.
        # FIXME: is this the right shape?
        if self.simulated:
            self.cam.grab_frame.return_value = \
                np.zeros((self.cfg.sensor_row_count,
                          self.cfg.sensor_column_count),
                         dtype=self.cfg.image_dtype)
            self.cam.print_statistics.return_value = "No simulated statistics."

    def setup_imaging_hardware(self):
        """provision the daq according to the dataset we are going to collect."""
        pass

    def check_system_memory_resources(self, channel_count: int,
                                      mem_chunk: int):
        """Make sure this machine can image under the specified configuration.

        :param channel_count: the number of channels we want to image with.
        :param mem_chunk: the number of images to hold in one chunk for
            compression
        :raises MemoryError:
        """
        # Calculate double buffer size for all channels.
        bytes_per_gig = (1024**3)
        used_mem_gigabytes = \
            ((self.cfg.bytes_per_image * mem_chunk * 2) / bytes_per_gig) \
            * channel_count
        # TODO: we probably want to throw in 1-2gigs of fudge factor.
        free_mem_gigabytes = virtual_memory()[1] / bytes_per_gig
        if free_mem_gigabytes < used_mem_gigabytes:
            raise MemoryError("System does not have enough memory to run "
                              "the specified number of channels. "
                              f"{used_mem_gigabytes:.1f}[GB] are required but "
                              f"{free_mem_gigabytes:.1f}[GB] are available.")

    def run_from_config(self):
        self.collect_volumetric_image(self.cfg.volume_x_um,
                                      self.cfg.volume_y_um,
                                      self.cfg.volume_z_um,
                                      self.cfg.channels,
                                      self.cfg.tile_overlap_x_percent,
                                      self.cfg.tile_overlap_y_percent,
                                      self.cfg.tile_prefix,
                                      self.cfg.compressor_chunk_size,
                                      self.cfg.local_storage_dir,
                                      # TODO: how to make these config based?
                                      self.img_storage_dir,
                                      self.deriv_storage_dir)

    # TODO: should z grid step size get passed in as a parameter rather than
    #   being read from the config?
    def collect_volumetric_image(self, volume_x_um: float, volume_y_um: float,
                                 volume_z_um: float,
                                 channels: list[int],
                                 tile_overlap_x_percent: float,
                                 tile_overlap_y_percent: float,
                                 tile_prefix: str,
                                 compressor_chunk_size: int = None,
                                 local_storage_dir: Path = Path("."),
                                 img_storage_dir: Path = None,
                                 deriv_storage_dir: Path = None):
        """Collect a tiled volumetric image with specified size/overlap specs.

        """
        mem_chunk = self.cfg.compressor_chunk_size \
            if compressor_chunk_size is None else compressor_chunk_size
        try:  # Ensure we have enough memory for the allocated chunk size.
            self.check_system_memory_resources(len(channels), mem_chunk)
        except MemoryError as e:
            self.log.error(e)
            raise

        # Compute: micrometers per grid step. At 0 tile overlap, this is just
        # the sensor's field of view.
        x_grid_step_um = \
            (1 - tile_overlap_x_percent/100.0) * self.cfg.tile_size_x_um
        y_grid_step_um = \
            (1 - tile_overlap_y_percent/100.0) * self.cfg.tile_size_y_um
        # Compute step and tile count.
        # Always round up so that we cover the desired imaging region.
        xsteps = ceil((volume_x_um - self.cfg.tile_size_x_um)
                      / x_grid_step_um)
        ysteps = ceil((volume_y_um - self.cfg.tile_size_y_um)
                      / y_grid_step_um)
        zsteps = ceil((volume_z_um - self.cfg.z_step_size_um)
                      / self.cfg.z_step_size_um)
        xtiles, ytiles, ztiles = (1 + xsteps, 1 + ysteps, 1 + zsteps)
        self.total_tiles = xtiles * ytiles * ztiles * len(channels)

        # Setup containers
        stack_transfer_workers = {}  # moves z-stacks to destination folder.
        # Reset the starting location.
        self.sample_pose.zero_in_place()
        self.stage_x_pos, self.stage_y_pos = (0, 0)
        # Iterate through the volume through z, then x, then y.
        # Play waveforms for the laser, camera trigger, and stage trigger.
        # Capture the fully-formed images as they arrive.
        # Create stacks of tiles along the z axis per channel.
        for y in range(ytiles):
            self.sample_pose.move_absolute(y=round(self.stage_y_pos),
                                           wait=True)
            self.stage_x_pos = 0
            for x in range(xtiles):
                self.sample_pose.move_absolute(x=round(self.stage_x_pos),
                                               wait=True)
                self.log.info(f"tile: ({x}, {y}); stage_position: "
                              f"({self.stage_x_pos:.3f}[um], "
                              f"{self.stage_y_pos:.3f}[um])")
                stack_prefix = f"{tile_prefix}_" \
                               f"{self.stage_x_pos}_{self.stage_y_pos}"
                # TODO: filename conventions across these functions must match.
                self._collect_tile_stacks(channels, ztiles, mem_chunk,
                                          stack_prefix, img_storage_dir,
                                          deriv_storage_dir)
                # Start transferring tiff file to its destination.
                # Note: Image transfer is faster than image capture.
                #   but we still wait for prior process to finish.
                if stack_transfer_workers:
                    self.log.info("Waiting for stack transfer processes "
                                  "to complete.")
                    for channel_name, worker in stack_transfer_workers.items():
                        worker.join()
                # Kick off Stack transfer processes per channel.
                # Bail early if we don't need to transfer anything.
                if not img_storage_dir or local_storage_dir == img_storage_dir:
                    self.log.info("Skipping file transfer process. File is "
                                  "already at its destination.")
                    continue
                for ch in channels:
                    filename = "FIXME"  # FIXME.
                    self.log.error(f"Starting transfer process for {filename}.")
                    filepath_src = local_storage_dir / filename
                    filepath_dest = img_storage_dir / filename
                    stack_transfer_workers[ch] = \
                        FileTransfer(filepath_src, filepath_dest,
                                     self.cfg.ftp, self.cfg.ftp_flags)
                # TODO: mip transfer processes.
                self.stage_x_pos += y_grid_step_um * UM_TO_STEPS
            self.stage_y_pos += self.cfg.z_step_size_um * UM_TO_STEPS

    def _collect_tile_stacks(self, channels: list[int], frame_count: int,
                             chunk_size: int, stack_prefix: str,
                             image_storage_dir: Path,
                             deriv_storage_dir: Path):
        """Collect tile stack for every specified channel and write them to
        disk compressed through ImarisWriter.

        The DAQ is already configured to play each laser and then move the
        stage for a specified amount of frames. This func simply collects all
        the images and then sends them to an external process (in chunks at a
        time) to compress them online and write them to disk.

        Since a single image can be ~300[MB], a stack of frames can
        easily be gigabytes.

        :param channels: a list of channels
        :param frame_count: number of frames to collect into a stack.
        :param chunk_size: the number of batch frames to send to
            the external compression process at a time.
        """
        # External processes.
        stack_writer_workers = {}  # writes img chunks to a stack on disk.
        # Put the backlash into a known state.
        stage_z_pos = 0
        z_backup_pos = -UM_TO_STEPS*self.cfg.stage_backlash_reset_dist_um
        self.log.debug("Applying extra move to take out backlash.")
        self.sample_pose.move_absolute(z=round(z_backup_pos), wait=True)
        self.sample_pose.move_absolute(z=stage_z_pos, wait=True)
        self.sample_pose.move_absolute(z=round(stage_z_pos), wait=True)
        self.setup_imaging_hardware()
        # Create storage containers (in ram) for a tile chunk per-channel.
        # TODO: change this to x,y,z if possible. Consider how it arrives
        #   from eGrabber.
        img_buffers = {}
        # TODO: we want xy, xz, and yz mips per channel
        mips = {}
        tile_name = {}
        for ch in channels:
            tile_name[ch] = f"{stack_prefix}_{ch}"
            # Note: a stack of 64 frames is ~18[GB] in memory.
            # FIXME: what is the shape?
            img_buffers[ch] = SharedDoubleBuffer((self.cfg.sensor_column_count,
                                                  self.cfg.sensor_row_count,
                                                  chunk_size),
                                                 dtype=self.cfg.datatype)
            mips[ch] = np.zeros((self.cfg.sensor_column_count,
                                 self.cfg.sensor_row_count),
                                dtype=self.cfg.datatype)
            # Create/Configure per-channel processes.
            self.log.debug(f"Creating StackWriter for {ch}[nm] channel.")
            stack_writer_workers[ch] = \
                StackWriter(self.cfg.sensor_row_count,
                            self.cfg.sensor_column_count,
                            frame_count, self.stage_x_pos, self.stage_y_pos,
                            self.cfg.x_voxel_size_um, self.cfg.y_voxel_size_um,
                            self.cfg.z_step_size_um,
                            self.cfg.compressor_chunk_size,
                            self.cfg.compressor_thread_count,
                            self.cfg.compressor_style,
                            self.cfg.datatype, self.img_storage_dir,
                            tile_name[ch], str(ch),
                            self.cfg.channel_specs[str(ch)]['hex_color'])
            stack_writer_workers[ch].start()
            self.mip_workers[ch] = MIPProcessor()  # has no configure()
        # data_logger is for the camera. It needs to exist between:
        #   cam.start() and cam.stop()
        #self.data_logger_worker = DataLogger(self.deriv_storage_dir,
        #                                     self.cfg.memento_path,
        #                                     f"{stack_prefix}_log")

        # For speed, data is transferred in chunks of a specific size.
        frame_num = 0  # The current frame we're on (channel agnostic).
        start_time = perf_counter()
        try:
            # TODO: wait for data_logger to actually start.
            #self.data_logger_worker.start()
            self.cam.start(live=False)
            self.ni.start()
            # We assume images arrive serialized in repeating channel order.
            last_frame_index = frame_count - 1
            for frame_index in range(frame_count):
                chunk_index = frame_index % chunk_size
                # Deserialize camera input into corresponding channel. i.e:
                # grab one tile per channel before moving onto the next tile.
                for ch_index in channels:
                    self.log.debug(f"Grabbing frame {frame_index} for "
                                   f"{ch_index}[nm] channel.")
                    # if self.simulated:
                    #     sleep(1/6.4)
                    # TODO: figure out how frombuffer works.
                    # TODO: don't do any reshaping if possible.
                    img_buffers[ch_index].write_buf[:, :, frame_index] = \
                        np.transpose(self.cam.grab_frame())
                    #self.cam.print_statistics(ch_index)
                # Dispatch chunk if we have aggregated a full chunk of frames.
                # OR if we are dispatching the last chunk, and it's not a
                # multiple of the chunk.
                if chunk_index == chunk_size - 1 or frame_index == last_frame_index:
                    # Wait for z stack writing to finish before dispatching
                    # more data.
                    while not all([w.done_reading.is_set() for _, w in stack_writer_workers.items()]):
                        sleep(0.001)
                    # Dispatch chunk to each stack-writing compression process.
                    # Toggle double buffer to continue writing images.
                    for ch_index in channels:
                        img_buffers[ch_index].toggle_buffers()
                        # Send over the read buffer shm name.
                        stack_writer_workers[ch_index].shm_name = \
                            img_buffers[ch_index].read_buf_mem_name
                        stack_writer_workers[ch_index].done_reading.clear()
        except Exception:
            raise
        finally:
            self.log.debug(f"Stack imaging time: {(perf_counter()-start_time)/3600.:.3f} hours.")
            self.log.debug("Closing devices and processes for this stack.")
            self.ni.stop()
            # self.ni.close()  # do we need this??
            self.cam.stop()
            # TODO: make sure this actually kills the data_logger
            #self.data_logger_worker.stop()
            #self.data_logger_worker.close()
            # Wait for stack writers to finish writing files to disk.
            for ch_name, worker in stack_writer_workers.items():
                self.log.debug(f"Closing {ch_name}[nm] channel StackWriter.")
                worker.join()
            # for ch_name, worker in self.mip_workers.items():
            #     self.log.debug(f"Closing {ch_name}[nm] channel Mip Worker.")
            #     worker.close()

            # Apply lead-in move to take out z backlash.
            z_backup_pos = -UM_TO_STEPS*self.cfg.stage_backlash_reset_dist_um
            self.log.debug("Applying extra move to take out backlash.")
            self.sample_pose.move_absolute(z=round(z_backup_pos), wait=True)
            self.sample_pose.move_absolute(z=0, wait=True)

            # Write MIPs to files.
            # TODO: do this outside of this function.
            # TODO: in the file_prefix, indicate if it is a XY, XZ, or YZ mip.
            #for ch, mip_data in mips.items():
            #    path = deriv_storage_dir/Path(f"{stack_prefix}_{ch}_mip.tiff")
            #    self.log.debug(f"Writing MIP for {ch} channel to: {path}")
            #    with TiffWriter(path, bigtiff=True) as tif:
            #        tif.write(mip_data)

    def _compute_mip_shapes(self, volume_x: float, volume_y: float,
                            volume_z: float, percent_x_overlap: float,
                            percent_y_overlap: float):
        """return three 2-tuples indicating the shapes of the mips."""
        raise NotImplementedError
        #xy, xz, yz = (0,0), (0,0), (0,0)

    def livestream(self):
        pass

    def close(self):
        """Safely close all open hardware connections."""
        # stuff here.
        self.log.info("Closing NIDAQ connection.")
        self.ni.close()
        # TODO: power down lasers.
        super().close()  # Call this last.
