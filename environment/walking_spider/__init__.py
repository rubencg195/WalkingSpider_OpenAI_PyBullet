import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='walkingspider-v0',
    entry_point='balance_bot.envs:WalkingSpiderEnv',
)