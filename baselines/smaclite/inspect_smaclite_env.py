import gymnasium as gym
import numpy as np
import smaclite  # noqa: F401


ENV_NAME = "smaclite/2s3z-v0"


def has(obj, name: str) -> bool:
    return hasattr(obj, name)


def safe_call(fn, *args, **kwargs):
    try:
        out = fn(*args, **kwargs)
        return ("OK", out)
    except Exception as e:
        return ("ERR", repr(e))


def main():
    env = gym.make(ENV_NAME)
    u = env.unwrapped

    print("=== ENV OBJECTS ===")
    print("env:", env)
    print("unwrapped:", u)

    print("\n=== BASIC ATTRS ===")
    for k in ["n_agents", "episode_limit"]:
        print(f"{k}: {getattr(u, k, None)}")

    print("\n=== HAS METHODS (SMAC-style) ===")
    methods = [
        "get_obs",
        "get_obs_agent",
        "get_obs_size",
        "get_state",
        "get_state_size",
        "get_avail_actions",
        "get_avail_agent_actions",
        "get_total_actions",
        "get_env_info",
    ]
    for m in methods:
        print(f"{m}: {has(u, m)}")

    print("\n=== GYMNASIUM SPACES ===")
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)

    print("\n=== PRE-RESET CALLS (should not crash wrapper init) ===")
    if has(u, "get_state_size"):
        print("get_state_size:", safe_call(u.get_state_size)[0])
    if has(u, "get_total_actions"):
        print("get_total_actions:", safe_call(u.get_total_actions)[0])
    if has(u, "get_obs_size"):
        print("get_obs_size:", safe_call(u.get_obs_size)[0])

    print("\n=== RESET ===")
    ob, info = env.reset()
    print("reset info keys:", list(info.keys()) if isinstance(info, dict) else type(info))
    print("type(ob):", type(ob))
    if isinstance(ob, (list, tuple)):
        print("len(ob):", len(ob))
        print("ob[0].shape:", np.asarray(ob[0]).shape)
    else:
        print("ob.shape:", np.asarray(ob).shape)

    print("\n=== POST-RESET CALLS (these define wrapper outputs) ===")
    if has(u, "get_state"):
        status, out = safe_call(u.get_state)
        print("get_state:", status, (np.asarray(out).shape if status == "OK" else out))

    if has(u, "get_avail_actions"):
        status, out = safe_call(u.get_avail_actions)
        arr = np.asarray(out) if status == "OK" else None
        print("get_avail_actions:", status, (arr.shape if status == "OK" else out))
        if status == "OK":
            # sanity: how many actions are available for agent 0
            print("avail[0] valid count:", int((arr[0] > 0).sum()))

    print("\n=== ONE STEP (VALID ACTIONS ONLY) ===")
    # sample valid actions from avail mask if available
    if has(u, "get_avail_actions"):
        avail = np.asarray(u.get_avail_actions())
        acts = []
        for i in range(avail.shape[0]):
            valid = np.flatnonzero(avail[i] > 0)
            acts.append(int(np.random.choice(valid)) if valid.size else 0)
    else:
        # fallback: sample from action_space (may be invalid in SMACLite!)
        acts = env.action_space.sample()

    out = env.step(acts)
    print("step returns len:", len(out))
    print("step types:", [type(x) for x in out])

    env.close()


if __name__ == "__main__":
    main()
