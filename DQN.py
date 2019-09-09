from stable_baselines.deepq.dqn import *
from stable_baselines.a2c.utils import conv, conv_to_fc, linear
from keras.layers import MaxPooling2D, AveragePooling2D

def nature_cnn_ex(scaled_images, **kwargs):
    """
    CNN from Nature paper.
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = conv(scaled_images, 'c1', n_filters=32, filter_size=7, stride=2, init_scale=np.sqrt(2), pad='SAME', **kwargs)
    layer_1 = activ(layer_1)
    layer_2 = conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs)
    layer_2 = conv(layer_2, 'c2_2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs)
    layer_2 = activ(tf.layers.batch_normalization(layer_2))
    layer_2 = tf.nn.max_pool(layer_2, (3, 2), (2, 2), "VALID")
    layer_3 = conv(layer_2, 'c31', n_filters=16, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs)
    layer_3 = conv(layer_3, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs)
    layer_3 = activ(layer_3)
    layer_3 = tf.nn.max_pool(layer_3, (2, 2), (2, 2), "VALID")
    layer_4 = conv(layer_3, 'c41', n_filters=16, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs)
    layer_4 = conv(layer_4, 'c4', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs)
    layer_4 = activ(tf.layers.batch_normalization(layer_4))
    layer_4 = tf.nn.avg_pool(layer_4, (7, 7), (1, 1), 'VALID')
    layer_4 = conv_to_fc(layer_4)
    return activ(linear(layer_4, 'fc1', n_hidden=128, init_scale=np.sqrt(2)))


def nature_cnn_ex2(scaled_images, **kwargs):
    """
    CNN from Nature paper.
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = conv(scaled_images, 'c1', n_filters=32, filter_size=8,  stride=4, init_scale=np.sqrt(2), pad='SAME', **kwargs)
    layer_1 = activ(layer_1)
    layer_2 = conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), pad='SAME', **kwargs)
    layer_2 = conv(layer_2, 'c2_1', n_filters=16, filter_size=1, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs)
    layer_2 = conv(layer_2, 'c2_2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs)
    layer_2 = activ(tf.layers.batch_normalization(layer_2))
    layer_2 = tf.nn.max_pool(layer_2, (3, 2), (2, 2), "VALID")
    layer_3 = conv(layer_2, 'c31', n_filters=32, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs)
    layer_3 = conv(layer_3, 'c3', n_filters=128, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs)
    layer_3 = activ(layer_3)
    layer_4 = tf.nn.avg_pool(layer_3, (4, 3), (1, 1), 'VALID')
    layer_4 = conv_to_fc(layer_4)
    return activ(linear(layer_4, 'fc1', n_hidden=128, init_scale=np.sqrt(2)))

class AtariDQN(DQN):

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=1, tb_log_name="DQN",
              reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(seed)

            # Create the replay buffer
            if self.prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = total_timesteps
                else:
                    prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                    initial_p=self.prioritized_replay_beta0,
                                                    final_p=1.0)
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.beta_schedule = None

            if replay_wrapper is not None:
                assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
                self.replay_buffer = replay_wrapper(self.replay_buffer)

            # Create the schedule for exploration starting from 1.
            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                              initial_p=1.0,
                                              final_p=self.exploration_final_eps)

            episode_rewards = [0.0]
            episode_successes = []
            obs = self.env.reset()
            reset = True
            self.episode_reward = np.zeros((1,))

            for _ in range(total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break
                # Take action and update exploration to the newest value
                kwargs = {}
                if not self.param_noise:
                    update_eps = self.exploration.value(self.num_timesteps)
                    update_param_noise_threshold = 0.
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = \
                        -np.log(1. - self.exploration.value(self.num_timesteps) +
                                self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True
                with self.sess.as_default():
                    action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                env_action = action
                reset = False
                new_obs, rew, done, info = self.env.step(env_action)
                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

                if writer is not None:
                    ep_rew = np.array([rew]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                                                                      self.num_timesteps)

                episode_rewards[-1] += rew
                if done:
                    for train_intense in range(256):
                        can_sample = self.replay_buffer.can_sample(self.batch_size)
                        if can_sample and self.num_timesteps > self.learning_starts:
                            if self.prioritized_replay:
                                experience = self.replay_buffer.sample(self.batch_size,
                                                                       beta=self.beta_schedule.value(
                                                                           self.num_timesteps))
                                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                            else:
                                obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                                weights, batch_idxes = np.ones_like(rewards), None

                            if writer is not None:
                                if (1 + self.num_timesteps) % 100 == 0:
                                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                    run_metadata = tf.RunMetadata()
                                    summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1,
                                                                          obses_tp1,
                                                                          dones, weights, sess=self.sess,
                                                                          options=run_options,
                                                                          run_metadata=run_metadata)
                                    writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                                else:
                                    summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1,
                                                                          obses_tp1,
                                                                          dones, weights, sess=self.sess)
                                writer.add_summary(summary, self.num_timesteps)
                            else:
                                _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones,
                                                                weights,
                                                                sess=self.sess)

                            if self.prioritized_replay:
                                new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                                self.replay_buffer.update_priorities(batch_idxes, new_priorities)

                            if can_sample and self.num_timesteps > self.learning_starts and \
                                    train_intense % self.target_network_update_freq == 0:
                                self.update_target(sess=self.sess)

                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)
                    reset = True

                    if num_episodes % 100 == 0:
                        self.save('/model/{}_{}_ep'.format(self.env.game_id, num_episodes+9600))

                if len(episode_rewards[-101:-1]) == 0:
                    mean_100ep_reward = -np.inf
                else:
                    mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    logger.record_tabular("steps", self.num_timesteps)
                    logger.record_tabular("episodes", num_episodes)
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("episode reward", episode_rewards[-2])
                    logger.record_tabular("% time spent exploring",
                                          int(100 * self.exploration.value(self.num_timesteps)))
                    logger.dump_tabular()

                self.num_timesteps += 1

        return self