import tensorflow as tf
import baselines.common.tf_util as U

def comp_Q(q_values,num_actions):
    Qi = []
    z = tf.expand_dims(tf.linspace(-10.0,10.0,51),0)
    for i in range(num_actions):
#        extheta = tf.exp(q_values[i])
#        sumtheta = tf.reduce_sum(extheta)
#        p = extheta/sumtheta
        result = tf.matmul(z,tf.transpose(q_values[i]))
        Qi.append(tf.reduce_sum(result,1))
#    Qi = tf.reduce_sum(tf.convert_to_tensor(Qi),axis=1)
    return Qi
   


def scope_vars(scope, trainable_only=False):
    """
    Get variables inside a scope
    The scope can be specified as a string
    Parameters
    ----------
    scope: str or VariableScope
        scope in which the variables reside.
    trainable_only: bool
        whether or not to return only the variables that were marked as trainable.
    Returns
    -------
    vars: [tf.Variable]
        list of variables in `scope`.
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )


def scope_name():
    """Returns the name of current scope as a string, e.g. deepq/q_func"""
    return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
    """Appends parent scope name to `relative_scope_name`"""
    return scope_name() + "/" + relative_scope_name


def build_act(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = make_obs_ph("observation")
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")
        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))
        q_values = q_func(inpt=observations_ph.get(), scope="q_func")
        Qi = comp_Q(q_values,num_actions)
        deterministic_actions = tf.argmax(Qi,0)
#        deterministic_actions = tf.sum_reduce(deterministic_actions,0)        

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        _act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        def act(ob, stochastic=True, update_eps=-1):
            return _act(ob, stochastic, update_eps)
        return act

def build_train(make_obs_ph, q_func, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0,
    scope="deepq", reuse=None):
    act_f = build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)
    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = make_obs_ph("obs_t")
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = make_obs_ph("obs_tp1")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q network evaluation
        q_t = q_func(inpt=obs_t_input.get(), scope="q_func", reuse=True) 
#        q_t = comp_Q(q_t,num_actions)
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func")

        # target q network evalution
        q_tp1 = q_func(inpt=obs_tp1_input.get(), scope="target_q_func")
#        q_tp1 = comp_Q(q_tp1,num_actions)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/target_q_func")

        # q scores for actions which we know were selected in the given state.
        a = tf.transpose(tf.one_hot(act_t_ph, num_actions))
        b = tf.stack([a]*51,2)
        q_t_selected = tf.reduce_sum(q_t * b, 1)

        # compute estimate of best possible value starting from state at t + 1

        q_tp1_using_online_net = q_func(inpt=obs_tp1_input.get(), scope="q_func", reuse=True)

        Qi = comp_Q(q_tp1,num_actions)
        deterministic_actions = tf.argmax(Qi)

     #   q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
     #   q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
        a1 = tf.transpose(tf.one_hot(deterministic_actions,num_actions))
        b1 = tf.stack([a1]*51,2)
        q_tp1_best = tf.reduce_sum(q_tp1 * b1,0)

        v_max = 10.0
        v_min = -10.0
        delta_z = (v_max-v_min) / float(51-1)
        m_prob = [np.zeros((batch_size, 51)) for i in range(batch_size) ]
        z = tf.expand_dims(tf.linspace(-10.0,10.0,51),0)
        for i in range(batch_size):
            for j in range(num_atom):
                Tz = min(v_max, max(v_min, reward[i] + gamma * (1.0-done_mask_ph[i]) * z[j]))
                bj = (Tz - v_min) / delta_z 
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[action[i]][int(m_l)] += q_tp1_best[deterministic_actions][i][j] * (m_u - bj)
                m_prob[action[i]][int(m_u)] += q_tp1_best[deterministic_actions][i][j] * (bj - m_l)
		
		

        q_t_selected_target = m_prob		 
#        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best
	
        # compute RHS of bellman equation
#        q_t_selected_target = tf.transpose(tf.stack([rew_t_ph])) + gamma * q_tp1_best_masked # 数+向量
        q_t_selected_target1 = tf.stop_gradient(q_t_selected_target)
        # compute the error (potentially clipped)
        error = -tf.reduce_sum(q_t_selected_target1*tf.log(q_t_selected))
#        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
#        errors = U.huber_loss(td_error)
#        weighted_error = tf.reduce_mean(importance_weights_ph * errors)

        # compute optimization op (potentially with gradient clipping)
#        if grad_norm_clipping is not None:
#            gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars)
#            for i, (grad, var) in enumerate(gradients):
#                if grad is not None:
#                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
#            optimize_expr = optimizer.apply_gradients(gradients)
#        else:
#            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)
        optimize_expr = optimizer.minimize(error, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=error,
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

        return act_f, train, update_target, {'q_values': q_values}
