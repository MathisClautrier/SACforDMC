
import numpy as np
import torch
import os
import random
import time
import json
import dmc2gym
from argparse import Namespace
import numpy as np

import utils
from logger import Logger
from replay_buffer import ReplayBuffer

from agent import SAC


def evaluate(env, agent, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            agent.actor.eval()
            while not done:
                if sample_stochastically:
                    action,_ = agent.sample_action(obs)
                else:
                    action,_ = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward

            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
            agent.actor.train()
        
        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)

def train(encoder,cfg):
    with open(cfg) as f:
        args_dic = json.load(f)
    args = Namespace(**args_dic)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        frame_skip=args.action_repeat
    )
    env.seed(args.seed)
    env = utils.FrameStack(env, k=args.frame_stack,img_size = args.img_size)

    # make directory
    ts = time.gmtime() 
    ts = time.strftime("%m-%d", ts)    
    env_name = args.domain_name + '-' + args.task_name
    exp_name = env_name + '-' + ts + '-im' + str(args.image_size) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed)  + '-' + args.encoder_type
    args.work_dir = args.work_dir + '/'  + exp_name

    utils.make_dir(args.work_dir)
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("work on :",device)

    action_shape = env.action_space.shape

    obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
    pre_aug_obs_shape = (3*args.frame_stack,args.pre_transform_image_size,args.pre_transform_image_size)

    replay_buffer = ReplayBuffer(
        obs_shape=(5*args.encoder_output,),
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
    )

    agent = SAC(obs_shape,action_shape,device,encoder,**args_dic)

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(args.num_train_steps):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, args.num_eval_episodes, L, step,args)
            if args.save_model:
                agent.save_curl(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            f = agent.actor.transform(obs).cpu().data.numpy()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            agent.actor.pi.train(False)
            action,f = agent.sample_action(obs)
            agent.actor.pi.train(True)

        # run training update
        if step >= args.init_steps:
            num_updates = 1 
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)
        next_f = agent.actor.transform(next_obs).cpu().data.numpy()
        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(f, action, reward, next_f, done_bool)

        obs = next_obs
        f = next_f
        episode_step += 1

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    train("cfg/cfg.json")