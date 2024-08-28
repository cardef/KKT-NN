import gym
import gym_anm
import time
import numpy as np
from tqdm import trange
from pickle import dump

def run():
    dataset = []
    env = gym.make("gym_anm:ANM6Easy-v0")
    env.reset()

    for i_episode in trange(100):
        observation = env.reset()
        rewards = 0
        for t in trange(1000):
            #env.render(mode='human') # 呈現 environment

            action = env.action_space.sample()
            aux = (env.next_vars(observation)/100)[[3,4]]
            observation, reward, done, info = env.step(action)
            #self.V[:, self.bus_slack_id] = 1+0j

            
            #env.render()
            #time.sleep(0.0)
            rewards += reward # 累計 reward
            X = np.concatenate((action/100, aux))
            y = (observation[[2,4, 6, 9,11,13]]/100)
            #y = np.append(y, done).astype(np.float32)
            
            if done:
                print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
                break
            dataset.append((X, y))

    env.close()
    return dataset


if __name__ == "__main__":
    dataset = run()
    dump(dataset, open("loads/dataset_loads.pkl", "wb"))