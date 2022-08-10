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

Name: Upscaler
Author: K4YT3X
Date Created: May 27, 2021
Last Modified: March 20, 2022
"""

import math
import multiprocessing
import os
import queue
import signal
import subprocess
import time
from multiprocessing.managers import ListProxy
from multiprocessing.sharedctypes import Synchronized

from loguru import logger
from PIL import Image, ImageChops, ImageStat
from realcugan_ncnn_vulkan_python import Realcugan
from realsr_ncnn_vulkan_python import Realsr
from srmd_ncnn_vulkan_python import Srmd
from waifu2x_ncnn_vulkan_python import Waifu2x
from .superres import SuperRes

# fixed scaling ratios supported by the algorithms
# that only support certain fixed scale ratios
ALGORITHM_FIXED_SCALING_RATIOS = {
    "waifu2x": [1, 2],
    "srmd": [2, 3, 4],
    "realsr": [4],
    "realcugan": [1, 2, 3, 4],
    "edsr": [2, 3, 4],
    "espcn": [2, 3, 4],
    "fsrcnn": [2, 3, 4],
    "lapsrn": [2, 4, 8],
}

ALGORITHM_CLASSES = {
    "waifu2x": Waifu2x,
    "srmd": Srmd,
    "realsr": Realsr,
    "realcugan": Realcugan,
    "superres": SuperRes,
}


class Upscaler(multiprocessing.Process):
    def __init__(
        self,
        instance_number: int,
        processing_queue: multiprocessing.Queue,
        processed_frames: ListProxy,
        pause: Synchronized,
    ) -> None:
        multiprocessing.Process.__init__(self)
        self.running = False
        self.instance_number = instance_number
        self.processing_queue = processing_queue
        self.processed_frames = processed_frames
        self.pause = pause
        
        # Determine the number of GPUs in the system
        # This is hard to do cross-vendor/platform - NVidia for now
        self.num_gpus = self.num_nvidia_gpus()

        signal.signal(signal.SIGTERM, self._stop)

    def run(self) -> None:
        self.running = True
        logger.opt(colors=True).info(
            f"Upscaler process <blue>{self.name}</blue> initiating"
        )
        processor_objects = {}
        while self.running is True:
            try:
                # pause if pause flag is set
                if self.pause.value is True:
                    time.sleep(0.1)
                    continue

                try:
                    # get new job from queue
                    (
                        frame_index,
                        (image0, image1),
                        (
                            output_width,
                            output_height,
                            noise,
                            difference_threshold,
                            algorithm,
                        ),
                    ) = self.processing_queue.get(False)

                    # destructure settings
                except queue.Empty:
                    time.sleep(0.1)
                    continue

                difference_ratio = 0
                if image0 is not None:
                    difference = ImageChops.difference(image0, image1)
                    difference_stat = ImageStat.Stat(difference)
                    difference_ratio = (
                        sum(difference_stat.mean)
                        / (len(difference_stat.mean) * 255)
                        * 100
                    )

                # if the difference is lower than threshold
                # skip this frame
                if difference_ratio < difference_threshold:

                    # make sure the previous frame has been processed
                    if frame_index > 0:
                        while self.processed_frames[frame_index - 1] is None:
                            time.sleep(0.1)

                    # make the current image the same as the previous result
                    self.processed_frames[frame_index] = self.processed_frames[
                        frame_index - 1
                    ]

                # if the difference is greater than threshold
                # process this frame
                else:
                    width, height = image1.size

                    # calculate required minimum scale ratio
                    output_scale = max(output_width / width, output_height / height)

                    # select the optimal algorithm scaling ratio to use
                    # split out the model parameter if the algorithm is SuperRes
                    algo_model = None
                    if algorithm.startswith("superres"):
                        algorithm, algo_model = algorithm.split("-", 2)
                        supported_scaling_ratios = sorted(
                            ALGORITHM_FIXED_SCALING_RATIOS[algo_model]
                        )
                    else:
                        supported_scaling_ratios = sorted(
                            ALGORITHM_FIXED_SCALING_RATIOS[algorithm]
                        )

                    remaining_scaling_ratio = math.ceil(output_scale)
                    scaling_jobs = []

                    # if the scaling ratio is 1.0
                    # apply the smallest scaling ratio available
                    if remaining_scaling_ratio == 1:
                        scaling_jobs.append(supported_scaling_ratios[0])
                    else:
                        while remaining_scaling_ratio > 1:
                            for ratio in supported_scaling_ratios:
                                if ratio >= remaining_scaling_ratio:
                                    scaling_jobs.append(ratio)
                                    remaining_scaling_ratio /= ratio
                                    break

                            else:
                                found = False
                                for i in supported_scaling_ratios:
                                    for j in supported_scaling_ratios:
                                        if i * j >= remaining_scaling_ratio:
                                            scaling_jobs.extend([i, j])
                                            remaining_scaling_ratio /= i * j
                                            found = True
                                            break
                                    if found is True:
                                        break

                                if found is False:
                                    scaling_jobs.append(supported_scaling_ratios[-1])
                                    remaining_scaling_ratio /= supported_scaling_ratios[
                                        -1
                                    ]

                    for job in scaling_jobs:

                        # select a processor object with the required settings
                        # create a new object if none are available
                        processor_object = processor_objects.get((algorithm, job))
                        if processor_object is None:
                            gpuid = self.instance_number % self.num_gpus
                            algo_params = dict (gpuid=gpuid, noise=noise, scale=job)
                            if algorithm == "superres":
                                algo_params.update(model=algo_model)
                            processor_object = ALGORITHM_CLASSES[algorithm](**algo_params)
                            processor_objects[(algorithm, job)] = processor_object

                        # process the image with the selected algorithm
                        image1 = processor_object.process(image1)

                    # downscale the image to the desired output size and
                    # save the image to disk
                    image1 = image1.resize((output_width, output_height), Image.LANCZOS)
                    self.processed_frames[frame_index] = image1

            # send exceptions into the client connection pipe
            except (SystemExit, KeyboardInterrupt):
                break

            except Exception as error:
                logger.exception(error)
                break

        logger.opt(colors=True).info(
            f"Upscaler process <blue>{self.name}</blue> terminating"
        )
        return super().run()

    def _stop(self, _signal_number, _frame) -> None:
        self.running = False
    
    def num_nvidia_gpus(self) -> int:
        try:
            p = subprocess.Popen(["nvidia-smi","--list-gpus"], stdout=subprocess.PIPE)
            stdout, stderror = p.communicate()
        except Exception as error:
            return 0
        
        output = stdout.decode('UTF-8')
        lines = output.split(os.linesep)
        num_devices = len(lines) - 1
        return num_devices
