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

Name: SuperRes
Author: K4YT3X
Date Created: May 27, 2021
Last Modified: March 20, 2022
"""

import cv2
import numpy as np
import PIL

class SuperRes:
    def __init__(
        self,
        gpuid: int = 0,
        noise: int = -1,
        scale: int = 2,
        model: str = "lapsrn",
        **kwargs,
    ) -> None:
        self.version = 1.0
        assert gpuid >= -1, "gpuid must be >= -1"
        assert noise in range(-1, 4), "noise must be 1-3"
        assert scale in [2, 3, 4, 8], "scale must be 2, 3, 4 or 8"
        assert model in ["edsr", "espcn", "fsrcnn", "lapsrn"], "model must be one of edsr, espcn, fsrcnn or lapsrn"
        
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel("video2x/models/{m}_x{s}.pb".format(m=model,s=scale))
        self.sr.setModel(model, scale)
        self.sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
    def process(self, image: PIL.Image) -> PIL.Image:
        # Convert from PIL to OpenCV
        cv2_image = np.array(image)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
        
        result = self.sr.upsample(cv2_image)
        
        # Convert back to PIL
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(result)