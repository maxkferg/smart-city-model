class config():
    # env config
    render_train     = False
    render_test      = False
    env_name         = "Pong-v0"
    overwrite_render = True
    record           = True
    high             = 255.

    # output config
    output_path  = "results/q5_train_atari_nature/"
    model_name   = "collision"
    model_output = output_path + "model.weights"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path + "monitor/"

    # model and training config
    num_episodes_test = 50
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 250000
    log_freq          = 100
    eval_freq         = 250000
    record_freq       = 250000
    soft_epsilon      = 0.05

    # nature paper hyper params
    nsteps_train       = 5*10**7
    batch_size         = 64
    buffer_size        = 5*10**6
    target_update_freq = 5000
    gamma              = 0.99
    learning_freq      = 1
    state_history      = 1
    skip_frame         = 4
    lr_begin           = 0.0010
    lr_end             = 0.0001
    lr_nsteps          = nsteps_train/2
    eps_begin          = 0.4
    eps_end            = 0.1
    eps_nsteps         = 5000000
    learning_start     = 50000
