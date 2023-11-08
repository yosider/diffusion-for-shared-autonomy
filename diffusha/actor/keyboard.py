import numpy as np
import pygame
from pygame.locals import (
    K_DOWN,
    K_ESCAPE,
    K_LEFT,
    K_RIGHT,
    K_UP,
    KEYDOWN,
    KEYUP,
    QUIT,
    RESIZABLE,
    VIDEORESIZE,
)

from diffusha.actor import Actor


class LunarLanderKeyboardActor(Actor):
    """Keyboard Controller for Lunar Lander."""

    def __init__(self, env, fps=50):
        """Init."""
        self.env = env
        self.human_agent_action = np.array([0.0, 0.0], dtype=np.float32)
        pygame.init()
        self.t = None
        self.fps = fps

    def _get_human_action(self):
        events = pygame.event.get()
        for event in events:
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    self.human_agent_action[0] = -1.0
                elif event.key == K_DOWN:
                    self.human_agent_action[0] = 1.0
                elif event.key == K_LEFT:
                    self.human_agent_action[1] = -1.0
                elif event.key == K_RIGHT:
                    self.human_agent_action[1] = 1.0
            elif event.type == KEYUP:
                if event.key == K_ESCAPE:
                    exit()
                elif event.key in [K_UP, K_DOWN]:
                    self.human_agent_action[0] = 0
                elif event.key in [K_LEFT, K_RIGHT]:
                    self.human_agent_action[1] = 0
            elif event.type == VIDEORESIZE:
                self.win = pygame.display.set_mode(event.size, RESIZABLE)
                self.size = event.size
            elif event.type == QUIT:
                exit()

        if abs(self.human_agent_action[1]) < 0.1:  # TODO: ???
            self.human_agent_action[0] = 0.0

        return self.human_agent_action

    def act(self, ob):
        """Act."""
        action = self._get_human_action()
        return action

    def reset(self):
        self.human_agent_action[:] = 0.0


if __name__ == "__main__":
    from diffusha.data_collection.env import make_env

    env = make_env("LunarLander-v1", seed=1, test=False)

    actor = LunarLanderKeyboardActor(env)

    for _ in range(10):
        ob = env.reset()
        env.render()
        done = False
        reward = 0.0

        while not done:
            env.render()
            ob, r, done, _ = env.step(actor.act(ob))
            reward += r
        print(reward)
