'''
ViZDoom wrapper
'''
import sys
import os

import vizdoom
import random
import time
import numpy as np
import re
import cv2

import vizdoom_utils.sensors as sensors


class DoomSimulator:
    def __init__(self, args):
        self.config = args['config']
        self.resolution = args['resolution']
        self.frame_skip = args['frame_skip']
        self.color_mode = args['color_mode']
        self.switch_maps = args['switch_maps']
        self.maps = args['maps']
        self.game_args = args['game_args']

        self.meas_to_predict = args['meas_to_predict']

        self._game = vizdoom.DoomGame()
        self._game.load_config(self.config)
        self._game.add_game_args(self.game_args)
        self.curr_map = 0
        self._game.set_doom_map(self.maps[self.curr_map])

        # set resolution
        try:
            self._game.set_screen_resolution(getattr(vizdoom.ScreenResolution, 'RES_%dX%d' % self.resolution))
            self.resize = False
        except:
            print("Requested resolution not supported:", sys.exc_info()[0], ". Setting to 160x120 and resizing")
            self._game.set_screen_resolution(getattr(vizdoom.ScreenResolution, 'RES_160X120'))
            self.resize = True

        # set color mode
        if self.color_mode == 'RGB':
            self._game.set_screen_format(vizdoom.ScreenFormat.CRCGCB)
        elif self.color_mode == 'GRAY':
            self._game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
        else:
            print("Unknown color mode")
            raise

        self.available_controls, self.continuous_controls, self.discrete_controls = self.analyze_controls(self.config)
        self.num_buttons = self._game.get_available_buttons_size()
        assert(self.num_buttons == len(self.discrete_controls) + len(self.continuous_controls))
        assert(len(self.continuous_controls) == 0) # only discrete for now
        self.num_meas = len(self.meas_to_predict)

        self.meas_tags = []
        for nm in range(self.num_meas):
            if nm in self.meas_to_predict:
                self.meas_tags.append('meas' + str(nm))

        self.episode_count = 0
        self.game_initialized = False

        self.predict_vision = args.get('predict_vision', False)
        self.sensors = sensors.VizdoomSensors.init_from_args(self._game, args['sensor_args'])
        self.num_channels = self.sensors.channels

    def analyze_controls(self, config_file):
        with open(config_file, 'r') as myfile:
            config = myfile.read()
        m = re.search('available_buttons[\s]*\=[\s]*\{([^\}]*)\}', config)
        avail_controls = m.group(1).split()
        cont_controls = np.array([bool(re.match('.*_DELTA', c)) for c in avail_controls])
        discr_controls = np.invert(cont_controls)
        return avail_controls, np.squeeze(np.nonzero(cont_controls)), np.squeeze(np.nonzero(discr_controls))

    def init_game(self):
        if not self.game_initialized:
            self._game.init()
            self.game_initialized = True

    def close_game(self):
        if self.game_initialized:
            self._game.close()
            self.game_initialized = False

    def step(self, action=0):
        """
        Action can be either the number of action or the actual list defining the action

        Args:
            action - action encoded either as an int (index of the action) or as a bool vector
        Returns:
            img  - image after the step
            meas - numpy array of returned additional measurements (e.g. health, ammo) after the step
            rwrd - reward after the step
            term - if the state after the step is terminal
        """
        self.init_game()

        rwrd = self._game.make_action(action, self.frame_skip)
        state = self._game.get_state()

        self.sensors.tick()

        if state is None:
            img = None
            meas = None
        else:
            # Input is (c, h, w).
            img = self.sensors.get_all(True)

            if not self.predict_vision:
                # To (h, w, c) for opencv.
                img = img.transpose((1, 2, 0))

                if self.resize:
                    img = cv2.resize(img, (self.resolution[0], self.resolution[1]))

                # Opencv squeezes (h, w, 1) -> (h, w)
                if img.ndim == 2:
                    img = img[:,:,np.newaxis]

                # Back to (h, w, c).
                img = img.transpose((2, 0, 1))

            # this is a numpy array of game variables specified by the scenario
            meas = [state.game_variables[i] for i in self.meas_to_predict]

        term = self._game.is_episode_finished() or self._game.is_player_dead()

        if term:
            # in multiplayer multi_simulator takes care of this
            self.new_episode()

             # should ideally put nans here, but since it's an int...
            img = np.zeros((self.num_channels, self.resolution[1], self.resolution[0]), dtype=np.float32)
            meas = np.zeros(self.num_meas, dtype=np.uint32)

        return img, meas, rwrd, term

    def get_random_action(self):
        return [(random.random() >= .5) for i in range(self.num_buttons)]

    def is_new_episode(self):
        return self._game.is_new_episode()

    def next_map(self):
        if self.switch_maps:
            self.curr_map = (self.curr_map+1) % len(self.maps)
            self._game.set_doom_map(self.maps[self.curr_map])

    def new_episode(self):
        self.next_map()
        self.episode_count += 1
        self._game.new_episode()
