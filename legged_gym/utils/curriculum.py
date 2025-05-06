curriculum_steps = [
                        [ #  ("rewards.scales.orientation", -1),
                        ("rewards.scales.base_height", 0),
                        ("rewards.scales.torques", -0.0002),
                        ("rewards.scales.dof_pos_limits", -10.0),
                        ("rewards.scales.tracking_lin_vel", 5),
                        ("rewards.scales.tracking_ang_vel", 5),
                        ("noise.noise_scales.lin_vel", 0.2),],
                        # ("rewards.scales.feet_air_time", 2),
                        [("domain_rand.randomize_mass", True),
                        ("domain_rand.randomize_inertia", True),
                        ("domain_rand.randomize_base_com", True),],
                        [("domain_rand.randomize_stiffness", True),
                        ("domain_rand.randomize_damping", True),
                        ("domain_rand.randomize_motor_strength", True),
                        ("domain_rand.randomize_motor_offset", True),],
                        [("domain_rand.randomize_gravity", True),
                        ("domain_rand.add_control_freq", True),
                        ("domain_rand.add_delay", True),],
                    ]

class CurriculumEnvManager:
    def __init__(self, env, steps=curriculum_steps):
        self.env = env
        self.steps = steps
        self.current_step = 0
        self.i = 0

    def step(self):
        if self.current_step < len(self.steps):
            attributes = self.steps[self.current_step]
            self.title = ""
            for attr_path, value in attributes:
                # Set attribute dynamically
                obj = self.env.cfg
                *parents, attr = attr_path.split(".")
                for parent in parents:
                    obj = getattr(obj, parent)
                self.title = f"{self.title}_{attr}"
                setattr(obj, attr, value)
                self.env._update_cfg(self.env.cfg)
                print(f"SET attribute {attr_path} to {value}. TRAINING.")

                if "rewards" in attr_path:
                    self.env.reward_scales[attr] = value
            return True
        else :
            print("Curriculum training completed.")
            return False
        
    def get_title(self):
        return self.title
    
    def apply_all_steps(self):
        while self.step():
            pass