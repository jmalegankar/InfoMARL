import torch
import gc
from tensordict import TensorDict
from benchmarl.algorithms import MappoConfig, QmixConfig, MasacConfig, IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig, Callback
from benchmarl.models.mlp import MlpConfig

class TrainingKeyInspector(Callback):
    def on_train_end(self, training_td: TensorDict, group: str):
        print(f"\n[{self.algo_name.upper()}] Training Keys Found:")
        # Print sorted list of keys to make it easy to read
        keys = sorted(training_td.keys(include_nested=True))
        for k in keys:
            print(f"  - {k}")
        print("-" * 40)

def run_debug():
    # We test 3 representative algorithms to see their unique loss keys
    algos = {
        "mappo": MappoConfig,
        "ippo": IppoConfig,
        "qmix": QmixConfig,
        "masac": MasacConfig
    }
    
    task = VmasTask.FOOD_COLLECTION.get_from_yaml()
    task.config["n_agents"] = 2
    task.config["max_steps"] = 10 

    model_config = MlpConfig.get_from_yaml()
    
    # Tiny config just to trigger one training step
    exp_config = ExperimentConfig.get_from_yaml()
    exp_config.max_n_frames = 200
    exp_config.on_policy_collected_frames_per_batch = 100
    exp_config.on_policy_n_envs_per_worker = 10
    exp_config.off_policy_collected_frames_per_batch = 100
    exp_config.off_policy_n_envs_per_worker = 10
    exp_config.off_policy_train_batch_size = 32
    exp_config.off_policy_memory_size = 500

    for name, AlgoClass in algos.items():
        print(f"\n>>> Checking {name.upper()}...")
        inspector = TrainingKeyInspector()
        inspector.algo_name = name # Hack to pass name to callback

        try:
            experiment = Experiment(
                task=task,
                algorithm_config=AlgoClass.get_from_yaml(),
                model_config=model_config,
                seed=0,
                config=exp_config,
                callbacks=[inspector]
            )
            experiment.run()
        except Exception as e:
            # We expect it to finish or crash after printing keys
            pass
        
        gc.collect()

if __name__ == "__main__":
    run_debug()