import tensorflow as tf
import baselines.common.tf_util as U
import numpy as np

def comp_Q(q_values,num_actions):
    Qi = []
    z = tf.expand_dims(tf.linspace(-10.0,10.0,51),0)
    for i in range(num_actions):
        result = tf.matmul(z,tf.transpose(q_values[i]))
#        print(q_values)
        Qi.append(tf.reduce_sum(result,0)) 
#    Qi = tf.argmax(tf.matmul(z,tf.transpose(q_values)),0)
#    Qi = tf.reduce_sum(Qi,axis=1)
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
      #  deterministic_actions = Qi
        deterministic_actions = tf.argmax(Qi,0)        

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



def build_deterministic_actions(make_obs_ph, q_func, num_actions,gamma=1.0,
    scope="deepq", reuse=None,batch_size=1):
    act_f = build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)
    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = make_obs_ph("obs_t")
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = make_obs_ph("obs_tp1")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        with tf.variable_scope("god"):
            m_prob = tf.get_variable("m", shape=[batch_size,51])
        # q network evaluation
        q_t = q_func(inpt=obs_t_input.get(), scope="q_func", reuse=True) 
#        q_t = comp_Q(q_t,num_actions)
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func")

        # target q network evalution
        q_tp1 = q_func(inpt=obs_tp1_input.get(), scope="target_q_func")
#        q_tp1 = comp_Q(q_tp1,num_actions)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/target_q_func")


        # compute estimate of best possible value starting from state at t + 1

        q_tp1_using_online_net = q_func(inpt=obs_tp1_input.get(), scope="q_func", reuse=True)

        Qi = comp_Q(q_tp1,num_actions)
        deterministic_actions = tf.argmax(Qi)

        a1 = tf.transpose(tf.one_hot(deterministic_actions,num_actions))
        b1 = tf.stack([a1]*51,2)
        q_tp1_best = tf.reduce_sum(q_tp1 * b1,0)

        v_max = 10.0
        v_min = -10.0
        delta_z = (v_max-v_min) / float(51-1)
        b1 = []
        b2 = []
        b = []
#        z = tf.expand_dims(tf.linspace(-10.0,10.0,51),0)
        z = tf.linspace(-10.0,10.0,51)
        for i in range(batch_size):
            c1 = []
            c2 = []
            c = []
            for j in range(51):
                Tz = tf.minimum(v_max, tf.maximum(v_min, rew_t_ph[i] + gamma * (1.0-done_mask_ph[i]) * z[j]))
                bj = (Tz - v_min) / delta_z 
                m_l, m_u = tf.to_int32(tf.floor(bj)), tf.to_int32(tf.ceil(bj))
                c1.append(m_l)
                c2.append(m_u)
                c.append(bj)
            b1.append(c1)
            b2.append(c2)
            b.append(c)
            

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        deterministic = U.function(
            inputs=[
			    obs_t_input,
                act_t_ph,
				rew_t_ph,
                obs_tp1_input,
				done_mask_ph,
            ],
            outputs=[
				deterministic_actions,
				act_t_ph,
				b1,
				b2,
				b,
				q_tp1_best,
			],  
        )
        update_target = U.function([], [], updates=[update_target_expr])
        return act_f, deterministic, update_target


def build_train(make_obs_ph, q_func, num_actions, optimizer, deterministic_pre, actions_pre, b_pre1, b_pre2, b_pre, q_tp1_best_pre, batch_size=1, grad_norm_clipping=None, gamma=1.0,
    scope="deepq", reuse=None,):
    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = make_obs_ph("obs_t")
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = make_obs_ph("obs_tp1")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")

        # q network evaluation
        q_t = q_func(inpt=obs_t_input.get(), scope="q_func", reuse=True) 
#        q_t = comp_Q(q_t,num_actions)
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func")

        # target q network evalution
        q_tp1 = q_func(inpt=obs_tp1_input.get(), scope="target_q_func",reuse=True)
#        q_tp1 = comp_Q(q_tp1,num_actions)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/target_q_func")

        # q scores for actions which we know were selected in the given state.
        a = tf.transpose(tf.one_hot(act_t_ph, num_actions))
        b = tf.stack([a]*51,2)
        q_t_selected = tf.reduce_sum(q_t * b, 0)

        # compute estimate of best possible value starting from state at t + 1

        q_tp1_using_online_net = q_func(inpt=obs_tp1_input.get(), scope="q_func", reuse=True)

        Qi = comp_Q(q_tp1_using_online_net,num_actions)
        deterministic_actions = tf.argmax(Qi)

     #   q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
     #   q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
        a1 = tf.transpose(tf.one_hot(deterministic_actions,num_actions))
        b1 = tf.stack([a1]*51,2)
        q_tp1_best = tf.reduce_sum(q_tp1 * b1,0)

        v_max = 10.0
        v_min = -10.0
        delta_z = (v_max-v_min) / float(51-1)
        m_prob = np.zeros([batch_size, 51])
        z = tf.expand_dims(tf.linspace(-10.0,10.0,51),0)
#        print(b_pre)
        for i in range(batch_size):
            for j in range(51):
#                m_prob[actions_pre[i]][b_pre1[i][j]] += q_tp1_best[deterministic_pre[i]][j] * (b_pre2[i][j] - b_pre[i][j])
#                m_prob[actions_pre[i]][b_pre2[i][j]] += q_tp1_best[deterministic_pre[i]][j] * (b_pre[i][j] - b_pre1[i][j])
#                print("(i,j)=(%d,%d),b_pre1ij=%d,b_pre2ij=%d" %(i,j,b_pre1[i][j],b_pre2[i][j]))
#                print("b_preij=%f" %b_pre[i][j])
#                print("deteri=%d" %deterministic_pre[i])
#                print(m_prob[i][b_pre1[i][j]])
#                print(m_prob[i][b_pre2[i][j]])
#                print(q_tp1_best[deterministic_pre[i]][j])
                m_prob[i][b_pre1[i][j]] += q_tp1_best_pre[deterministic_pre[i]][j] * (b_pre2[i][j] - b_pre[i][j])
                m_prob[i][b_pre2[i][j]] += q_tp1_best_pre[deterministic_pre[i]][j] * (b_pre[i][j] - b_pre1[i][j])
		
        with tf.variable_scope("god", reuse=True) as scope2:
            m_prob1 = tf.get_variable("m")
        tf.assign(m_prob1,m_prob)
        q_t_selected_target = tf.to_float(m_prob1)		 
#        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best
	
        # compute RHS of bellman equation
#        q_t_selected_target = tf.transpose(tf.stack([rew_t_ph])) + gamma * q_tp1_best_masked # 数+向量
        q_t_selected_target1 = tf.stop_gradient(q_t_selected_target)
        # compute the error (potentially clipped)
        error = -tf.reduce_sum(q_t_selected_target1*tf.log(q_t_selected))

        # compute optimization op (potentially with gradient clipping)
        optimize_expr = optimizer.minimize(error)


        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
            ],
            outputs=error,
            updates=[optimize_expr]
        )

        q_values = U.function([obs_t_input], q_t)

        return train, {'q_values': q_values}
