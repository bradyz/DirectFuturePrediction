import os
import time
import sys
import argparse

import numpy as np

import vizdoom_utils.sensors as sensors
import DFP.multi_experiment as multi_experiment



def additional_setup(simulator_args, agent_args, target_maker_args):
    _bool = lambda x: x.lower() in ['1', 'y']

    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('--enable_image', type=_bool, default=False)
    parser.add_argument('--enable_depth', type=_bool, default=False)
    parser.add_argument('--enable_label', type=_bool, default=False)
    parser.add_argument('--enable_flow', type=_bool, default=False)
    parser.add_argument('--enable_normal', type=_bool, default=False)
    parser.add_argument('--log_dir', type=str, required=True)

    args = parser.parse_args()

    sensor_args = sensors.SensorArguments(
            simulator_args['color_mode'],
            args.enable_image, args.enable_depth, args.enable_label,
            args.enable_flow, args.enable_normal)

    agent_args['sensor_args'] = sensor_args
    simulator_args['sensor_args'] = sensor_args

    enabled_strings = [
            'enable_image', 'enable_depth', 'enable_label', 'enable_flow', 'enable_normal'
            ]
    vector = list(map(lambda x: getattr(args, x), enabled_strings))

    agent_args['model_dir'] = '_'.join(map(lambda x: str(int(x)), vector))
    agent_args['log_dir'] = args.log_dir
    agent_args['checkpoint_dir'] = args.log_dir

    simulator_args['meas_to_predict'] = target_maker_args['meas_to_predict']
    simulator_args['config'] = (
            'maps/D3_battle.cfg' if args.mode == 'train' else
            'maps/D3_battle_test.cfg')

    print('Mode: %s, using %s.' % (args.mode, simulator_args['config']))
    print('Checkpoint path: %s.' % os.path.join(agent_args['checkpoint_dir'], agent_args['model_dir']))


def main(main_args):
    ## Target maker
    target_maker_args = {}
    target_maker_args['future_steps'] = [1,2,4,8,16,32]
    target_maker_args['meas_to_predict'] = [0,1,2]
    target_maker_args['min_num_targs'] = 3
    target_maker_args['rwrd_schedule_type'] = 'exp'
    target_maker_args['gammas'] = []
    target_maker_args['invalid_targets_replacement'] = 'nan'

    ## Simulator
    simulator_args = {}
    simulator_args['resolution'] = (84,84)
    simulator_args['frame_skip'] = 4
    simulator_args['color_mode'] = 'GRAY'
    simulator_args['use_shaping_reward'] = False
    simulator_args['maps'] = ['MAP01']
    simulator_args['switch_maps'] = False
    #train
    simulator_args['num_simulators'] = 8

    ## Experience
    # Train experience
    train_experience_args = {}
    train_experience_args['memory_capacity'] = 10000
    train_experience_args['history_length'] = 1
    train_experience_args['history_step'] = 1
    train_experience_args['action_format'] = 'enumerate'
    train_experience_args['shared'] = False

    # Test prediction experience
    test_prediction_experience_args = train_experience_args.copy()
    test_prediction_experience_args['memory_capacity'] = 1

    # Test policy experience
    test_policy_experience_args = train_experience_args.copy()
    test_policy_experience_args['memory_capacity'] = 5500

    ## Agent
    agent_args = {}

    # agent type
    agent_args['agent_type'] = 'advantage'

    # preprocessing
    agent_args['preprocess_input_measurements'] = lambda x: x / 100. - 0.5
    targ_scale_coeffs = np.expand_dims((np.expand_dims(np.array([7.5,30.,1.]),1) * np.ones((1,len(target_maker_args['future_steps'])))).flatten(),0)
    agent_args['preprocess_input_targets'] = lambda x: x / targ_scale_coeffs
    agent_args['postprocess_predictions'] = lambda x: x * targ_scale_coeffs

    # agent properties
    agent_args['objective_coeffs_temporal'] = [0., 0. ,0. ,0.5, 0.5, 1.]
    agent_args['objective_coeffs_meas'] = [0.5, 0.5, 1.]
    agent_args['random_exploration_schedule'] = lambda step: (0.02 + 145000. / (float(step) + 150000.))
    agent_args['new_memories_per_batch'] = 8

    # net parameters
    agent_args['conv_params']     = np.array([(32,8,4), (64,4,2), (64,3,1)],
                                     dtype = [('out_channels',int), ('kernel',int), ('stride',int)])
    agent_args['fc_img_params']   = np.array([(512,)], dtype = [('out_dims',int)])
    agent_args['fc_meas_params']  = np.array([(128,), (128,), (128,)], dtype = [('out_dims',int)]) 
    agent_args['fc_joint_params'] = np.array([(512,), (-1,)], dtype = [('out_dims',int)]) # we put -1 here because it will be automatically replaced when creating the net
    agent_args['weight_decay'] = 0.00000

    # optimization parameters
    agent_args['batch_size'] = 64
    agent_args['init_learning_rate'] = 0.0001
    agent_args['lr_step_size'] = 250000
    agent_args['lr_decay_factor'] = 0.3
    agent_args['adam_beta1'] = 0.95
    agent_args['adam_epsilon'] = 1e-4
    agent_args['optimizer'] = 'Adam'
    agent_args['reset_iter_count'] = False

    # directories
    agent_args['checkpoint_dir'] = 'checkpoints'
    agent_args['log_dir'] = 'logs'
    agent_args['init_model'] = ''
    agent_args['model_name'] = "predictor.model"
    agent_args['model_dir'] = time.strftime("%Y_%m_%d_%H_%M_%S")

    # logging and testing
    agent_args['print_err_every'] = 50
    agent_args['detailed_summary_every'] = 1000
    agent_args['test_pred_every'] = 0
    agent_args['test_policy_every'] = 7812
    agent_args['num_batches_per_pred_test'] = 0
    agent_args['num_steps_per_policy_test'] = test_policy_experience_args['memory_capacity'] / simulator_args['num_simulators']
    agent_args['checkpoint_every'] = 1000
    agent_args['save_param_histograms_every'] = 5000
    agent_args['test_policy_in_the_beginning'] = True

    # experiment arguments
    experiment_args = {}
    experiment_args['num_train_iterations'] = 820000
    experiment_args['test_objective_coeffs_temporal'] = np.array([0., 0., 0., 0.5, 0.5, 1.])
    experiment_args['test_objective_coeffs_meas'] = np.array([0.5,0.5,1.])
    experiment_args['test_random_prob'] = 0.
    experiment_args['test_checkpoint'] = 'checkpoints'
    experiment_args['test_policy_num_steps'] = 2000
    experiment_args['show_predictions'] = False
    experiment_args['multiplayer'] = False

    additional_setup(simulator_args, agent_args, target_maker_args)

    # Create and run the experiment
    experiment = multi_experiment.MultiExperiment(
            target_maker_args=target_maker_args,
            simulator_args=simulator_args,
            train_experience_args=train_experience_args,
            test_policy_experience_args=test_policy_experience_args,
            agent_args=agent_args,
            experiment_args=experiment_args)

    experiment.run(main_args[0])


if __name__ == '__main__':
    main(sys.argv[1:])
