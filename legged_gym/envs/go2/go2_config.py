from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        # FL_0,FL_1,FL_2,FR_0,FR_1,FR_2,RL_0,RL_1,RL_2,RR_0,RR_1,RR_2
        # hip = 0, thigh = 1, calf = 2
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class env(LeggedRobotCfg.env):
        num_observations = 48 # - 3 - 3

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_mass = True
        limb_mass_change_percent = 0.2 # 10%
        randomize_inertia = True
        intertia_change_percent = 0.5 # 10%
        push_robots = True
        push_interval_s = 7.5
        max_push_vel_xy = 2.

        add_control_freq = True
        randomize_control_freq_lambda = 500
        add_delay = True
        randomize_delay = [0, 8]

        randomize_stiffness = True
        randomize_stiffness_range = [0.7, 1.3]
        randomize_damping = True
        randomize_damping_range = [0.7, 1.3]


    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class noise( LeggedRobotCfg.noise ):
        # add_noise = True
        # noise_level = 1.0 # scales other values
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            # dof_pos = 0.01
            # dof_vel = 1.5
            lin_vel = 0.2
            # ang_vel = 0.2
            # gravity = 0.05
            # height_measurements = 0.1
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0
            alive = 1
            dof_vel_limits = -0.25
            feet_air_time = 5
            # stand_still = 5
            tracking_lin_vel = 2.5
            tracking_ang_vel = 1.5

        soft_dof_vel_limit = 0.017395 * 10
        #     lin_vel_z = 0 # requires base_lin_vel[:,2], don't have so must be 0 weight
        #     # ang_vel_xy = 0 # requires base_ang_vel[:,:2], default is -0.05
        #     # orientation = 0 # requires projected_gravity, it is 0 anyways in base config
        #     base_height = -175 # scale might be very off
        #     tracking_lin_vel = 0
        #     tracking_ang_vel = 0
        #     feet_air_time = 0
        #     stumble = 0
        #     stand_still = 0
        #     alive = 15
        #     # feet_contact_forces = 0
        #     dof_vel_limits = -2
            
        # soft_dof_vel_limit = 0.017395 * 2
        
class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'

  
