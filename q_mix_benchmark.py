import xuance

runner = xuance.get_runner(method='QMIX', env="mpe", env_id="simple_spread_v3", config_path='q_mix.yaml')
runner.run()