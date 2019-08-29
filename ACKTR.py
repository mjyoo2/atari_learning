from stable_baselines.acktr.acktr_disc import *

class checkpoint_ACKTR(ACKTR):
    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="ACKTR",
              reset_num_timesteps=True):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(seed)
            self.n_batch = self.n_envs * self.n_steps

            self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=total_timesteps,
                                                    schedule=self.lr_schedule)

            # FIFO queue of the q_runner thread is closed at the end of the learn function.
            # As a result, it needs to be redefinied at every call
            with self.graph.as_default():
                with tf.variable_scope("kfac_apply", reuse=self.trained,
                                       custom_getter=tf_util.outer_scope_getter("kfac_apply")):
                    # Some of the variables are not in a scope when they are create
                    # so we make a note of any previously uninitialized variables
                    tf_vars = tf.global_variables()
                    is_uninitialized = self.sess.run([tf.is_variable_initialized(var) for var in tf_vars])
                    old_uninitialized_vars = [v for (v, f) in zip(tf_vars, is_uninitialized) if not f]

                    self.train_op, self.q_runner = self.optim.apply_gradients(list(zip(self.grads_check, self.params)))

                    # then we check for new uninitialized variables and initialize them
                    tf_vars = tf.global_variables()
                    is_uninitialized = self.sess.run([tf.is_variable_initialized(var) for var in tf_vars])
                    new_uninitialized_vars = [v for (v, f) in zip(tf_vars, is_uninitialized)
                                              if not f and v not in old_uninitialized_vars]

                    if len(new_uninitialized_vars) != 0:
                        self.sess.run(tf.variables_initializer(new_uninitialized_vars))

            self.trained = True

            runner = A2CRunner(self.env, self, n_steps=self.n_steps, gamma=self.gamma)
            self.episode_reward = np.zeros((self.n_envs,))

            t_start = time.time()
            coord = tf.train.Coordinator()
            if self.q_runner is not None:
                enqueue_threads = self.q_runner.create_threads(self.sess, coord=coord, start=True)
            else:
                enqueue_threads = []

            # Training stats (when using Monitor wrapper)
            ep_info_buf = deque(maxlen=100)

            for update in range(1, total_timesteps // self.n_batch + 1):
                # true_reward is the reward without discount
                obs, states, rewards, masks, actions, values, ep_infos, true_reward = runner.run()
                ep_info_buf.extend(ep_infos)
                policy_loss, value_loss, policy_entropy = self._train_step(obs, states, rewards, masks, actions, values,
                                                                           self.num_timesteps // (self.n_batch + 1),
                                                                           writer)
                n_seconds = time.time() - t_start
                fps = int((update * self.n_batch) / n_seconds)

                if writer is not None:
                    self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                      true_reward.reshape((self.n_envs, self.n_steps)),
                                                                      masks.reshape((self.n_envs, self.n_steps)),
                                                                      writer, self.num_timesteps)

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                if update % 1000 == 0:
                    self.save('/model/{}_{}_ACKTR'.format('Breakout', update))

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, rewards)
                    logger.record_tabular("nupdates", update)
                    logger.record_tabular("total_timesteps", self.num_timesteps)
                    logger.record_tabular("fps", fps)
                    logger.record_tabular("policy_entropy", float(policy_entropy))
                    logger.record_tabular("policy_loss", float(policy_loss))
                    logger.record_tabular("value_loss", float(value_loss))
                    logger.record_tabular("explained_variance", float(explained_var))
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.dump_tabular()

                self.num_timesteps += self.n_batch + 1

            coord.request_stop()
            coord.join(enqueue_threads)

        return self