#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2018-2022 K4YT3X and contributors.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Name: Video Decoder
Author: K4YT3X
Date Created: June 17, 2021
Last Modified: March 21, 2022
"""

import contextlib
import multiprocessing
import os
import pathlib
import queue
import signal
import subprocess
import threading
import time
from multiprocessing.sharedctypes import Synchronized

import ffmpeg
from loguru import logger
from PIL import Image

from .pipe_printer import PipePrinter

# map Loguru log levels to FFmpeg log levels
LOGURU_FFMPEG_LOGLEVELS = {
    "trace": "trace",
    "debug": "debug",
    "info": "info",
    "success": "info",
    "warning": "warning",
    "error": "error",
    "critical": "fatal",
}


class VideoDecoder(threading.Thread):
    def __init__(
        self,
        input_path: pathlib.Path,
        input_width: int,
        input_height: int,
        frame_rate: float,
        processing_queue: multiprocessing.Queue,
        processing_settings: tuple,
        pause: Synchronized,
        deinterlace=False,
        ignore_max_image_pixels=True,
    ) -> None:
        threading.Thread.__init__(self)
        self.running = False
        self.input_path = input_path
        self.input_width = input_width
        self.input_height = input_height
        self.processing_queue = processing_queue
        self.processing_settings = processing_settings
        self.pause = pause

        # this disables the "possible DDoS" warning
        if ignore_max_image_pixels:
            Image.MAX_IMAGE_PIXELS = None

        self.exception = None
        pipeline = ffmpeg.input(input_path, r=frame_rate)["v"]
        if deinterlace :
            pipeline = pipeline.filter('yadif')
        self.decoder = subprocess.Popen(
            ffmpeg.compile(
                pipeline                
                .output("pipe:1", format="rawvideo", pix_fmt="rgb24", vsync="cfr")
                .global_args("-hide_banner")
                .global_args("-nostats")
                .global_args("-nostdin")
                .global_args(
                    "-loglevel",
                    LOGURU_FFMPEG_LOGLEVELS.get(
                        os.environ.get("LOGURU_LEVEL", "INFO").lower()
                    ),
                ),
                overwrite_output=True,
            ),
            env=dict(AV_LOG_FORCE_COLOR="TRUE", **os.environ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # start the PIPE printer to start printing FFmpeg logs
        self.pipe_printer = PipePrinter(self.decoder.stderr)
        self.pipe_printer.start()

    def run(self) -> None:
        self.running = True

        # the index of the frame
        frame_index = 0

        # create placeholder for previous frame
        # used in interpolate mode
        previous_image = None

        # continue running until an exception occurs
        # or all frames have been decoded
        while self.running is True:

            # pause if pause flag is set
            if self.pause.value is True:
                time.sleep(0.1)
                continue

            try:
                buffer = self.decoder.stdout.read(
                    3 * self.input_width * self.input_height
                )

                # source depleted (decoding finished)
                # after the last frame has been decoded
                # read will return nothing
                if len(buffer) == 0:
                    self.stop()
                    continue

                # convert raw bytes into image object
                image = Image.frombytes(
                    "RGB", (self.input_width, self.input_height), buffer
                )

                # keep checking if the running flag is set to False
                # while waiting to put the next image into the queue
                while self.running is True:
                    with contextlib.suppress(queue.Full):
                        self.processing_queue.put(
                            (
                                frame_index,
                                (previous_image, image),
                                self.processing_settings,
                            ),
                            timeout=0.1,
                        )
                        break

                previous_image = image
                frame_index += 1

            # most likely "not enough image data"
            except ValueError as error:
                self.exception = error

                # ignore queue closed
                if "is closed" not in str(error):
                    logger.exception(error)
                break

            # send exceptions into the client connection pipe
            except Exception as error:
                self.exception = error
                logger.exception(error)
                break
        else:
            logger.debug("Decoding queue depleted")

        # flush the remaining data in STDOUT and STDERR
        self.decoder.stdout.flush()
        self.decoder.stderr.flush()

        # send SIGINT (2) to FFmpeg
        # this instructs it to finalize and exit
        # On Windows, we use the keyboard interrupt instead of SIGINT
        if hasattr(signal, 'CTRL_C_EVENT'):
            os.kill(self.decoder.pid, signal.CTRL_C_EVENT)
        else:
            self.decoder.send_signal(signal.SIGINT)

        # close PIPEs to prevent process from getting stuck
        self.decoder.stdout.close()
        self.decoder.stderr.close()

        # wait for process to exit
        self.decoder.wait()

        # wait for PIPE printer to exit
        self.pipe_printer.stop()
        self.pipe_printer.join()

        logger.info("Decoder thread exiting")
        return super().run()

    def stop(self) -> None:
        self.running = False
