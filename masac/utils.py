import torch
import random
import numpy as np
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(checkpoint_dir, actor, critic, target_critic, actor_optimizer, critic_optimizer, 
                   alpha, alpha_optimizer, replay_buffer, global_step, update_step, best_reward=None):
    """
    Save a checkpoint of the training state.
    """
    checkpoint = {
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'target_critic_state_dict': target_critic.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
        'alpha': alpha.item(),
        'alpha_optimizer_state_dict': alpha_optimizer.state_dict(),
        'replay_buffer': replay_buffer.get_save_state(),
        'global_step': global_step,
        'update_step': update_step,
        'seed': seed,
        'best_reward': best_reward
    }
    
    # Regular checkpoint
    checkpoint_filename = os.path.join(checkpoint_dir, f'checkpoint_step_{global_step}.pt')
    torch.save(checkpoint, checkpoint_filename)
    print(f"Checkpoint saved to {checkpoint_filename}")
    
    # Always save the latest checkpoint for easy resume
    latest_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_checkpoint)
    
    # If this is a best reward checkpoint, save it separately
    if best_reward is not None:
        best_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_checkpoint)
        print(f"New best reward {best_reward:.2f} - saved best checkpoint")


def load_checkpoint(checkpoint, seed, actor, critic, target_critic, actor_optimizer,
                    critic_optimizer, alpha, alpha_optimizer, replay_buffer, device):
    """
    Load a checkpoint and restore the training state.
    """
    # Check if the seed matches
    if checkpoint['seed'] != seed:
        raise ValueError(f"Checkpoint seed {seed} does not match the current seed {config.SEED}.")

    # Restore the state
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
    actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
    alpha.copy_(checkpoint['alpha'])
    replay_buffer.load_state(checkpoint['replay_buffer'])
    global_step = checkpoint['global_step']
    update_step = checkpoint['update_step']
    best_reward = checkpoint.get('best_reward', None)

    return global_step, update_step, best_reward