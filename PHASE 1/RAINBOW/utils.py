def shape_reward(obs, next_obs, reward):

    shaped = reward

    if sum(next_obs[:16]) > 0:
        shaped += 1

    if next_obs[16] == 1:
        shaped += 5

    if next_obs[17] == 1:
        shaped += 15

    if sum(next_obs[:16]) == 0:
        shaped -= 0.5

    return shaped