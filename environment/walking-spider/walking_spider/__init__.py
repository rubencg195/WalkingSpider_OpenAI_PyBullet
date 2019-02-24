import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='WalkingSpider-v0',
    entry_point='walking_spider.envs:WalkingSpiderEnv',
)