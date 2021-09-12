#!/usr/bin/env python3
import os
from segment_utils import Segment
from typing import Optional

import numpy as np


from aido_schemas import EpisodeStart, protocol_agent_DB20, PWMCommands, DB20Commands, LEDSCommands, RGB, \
    wrap_direct, Context, DB20Observations, JPGImage, logger


from wrappers import DTPytorchWrapper
from PIL import Image
import io
from segment import start_segment
from segment_utils import Segment


class PytorchRLTemplateAgent:
    def __init__(self, load_model: bool, model_path: Optional[str]):
        Segment()
        self.load_model = load_model
        self.model_path = model_path
        self.segment_model = Segment.model
        self.tf_segment = Segment.test_transform


    def init(self, context: Context):
        self.check_gpu_available(context)
        logger.info('PytorchRLTemplateAgent init')
        from model import DDPG
        self.preprocessor = DTPytorchWrapper()

        self.model = DDPG(state_dim=self.preprocessor.shape, action_dim=2, max_action=1, net_type="cnn")
        self.current_image = np.zeros((640, 480, 3))

        if self.load_model:
            logger.info('Pytorch Template Agent loading models')
            fp = self.model_path if self.model_path else "model"
            self.model.load(fp, "models", for_inference=True)
        logger.info('PytorchRLTemplateAgent init complete')

    def check_gpu_available(self,context: Context):
        import torch
        available = torch.cuda.is_available()
        req = os.environ.get('AIDO_REQUIRE_GPU', None)
        context.info(f'torch.cuda.is_available = {available!r} AIDO_REQUIRE_GPU = {req!r}')
        context.info('init()')
        if available:
            i = torch.cuda.current_device()
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(i)
            context.info(f'device {i} of {count}; name = {name!r}')
        else:
            if req is not None:
                msg = 'I need a GPU; bailing.'
                context.error(msg)
                raise RuntimeError(msg)


    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}".')

    def on_received_observations(self, data: DB20Observations):
        camera: JPGImage = data.camera
        obs = jpg2rgb(camera.jpg_data)
        self.current_image = start_segment(obs, self.tf_segment, self.segment_model).transpose(2,0,1) #self.preprocessor.preprocess(obs)

    def compute_action(self, observation):
        #if observation.shape != self.preprocessor.transposed_shape:
        #    observation = self.preprocessor.preprocess(observation)
        #observation = start_segment(observation, self.tf_segment, self.segment_model)
        action = self.model.predict(observation)
        #return [1., 1.]
        print(action)
        return [0.3, 0.3]#action.astype(float)

    def on_received_get_commands(self, context: Context):
        pwm_left, pwm_right = self.compute_action(self.current_image)

        pwm_left = float(np.clip(pwm_left, -1, +1))
        pwm_right = float(np.clip(pwm_right, -1, +1))

        grey = RGB(0.0, 0.0, 0.0)
        led_commands = LEDSCommands(grey, grey, grey, grey, grey)
        pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
        commands = DB20Commands(pwm_commands, led_commands)
        context.write('commands', commands)

    def finish(self, context: Context):
        context.info('finish()')


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """ Reads JPG bytes as RGB"""
    im = Image.open(io.BytesIO(image_data))
    im = im.convert('RGB')
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data

def main():
    node = PytorchRLTemplateAgent(load_model=False, model_path=None)
    protocol = protocol_agent_DB20
    wrap_direct(node=node, protocol=protocol)


if __name__ == '__main__':
    main()
