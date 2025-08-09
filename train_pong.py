import torch
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import numpy as np
import time
import random
import os
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from typing import Tuple, Optional, Union
from mamba_dqn_network import AtariMambaDQNNetwork
from replay_buffer import PrioritizedReplayBufferGPU

SELECTED_MODEL_CLASS = AtariMambaDQNNetwork
MODEL_CLASS_NAME = SELECTED_MODEL_CLASS.__name__
AnyMambaDQNNetwork = Union[AtariMambaDQNNetwork]

ENV_NAME = 'ALE/Pong-v5'
SEQUENCE_LENGTH = 4

BUFFER_SIZE = 50000
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_STEPS = 900000
TARGET_UPDATE_FREQ = 10000
LEARNING_RATE = 1e-4
TOTAL_TRAINING_STEPS = 1500000
MIN_STEPS_FOR_TRAINING = 10000
ALPHA = 0.6
BETA_START = 0.4
LOG_INTERVAL = 10000
SAVE_INTERVAL = 1000000
SEED = 42

GRAD_CLIP_CONFIG = {
    "AtariMambaDQNNetwork": 1.0,
    "default": 1.0
}
MODEL_SAVE_DIR = f"saved_models_atari_{MODEL_CLASS_NAME.lower()}"


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_env(env_name_str: str) -> gym.Env:
    env = gym.make(env_name_str, render_mode=None)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=(84, 84))
    return env


def linear_epsilon_decay(step: int, start_eps: float, end_eps: float, decay_steps: int) -> float:
    fraction = min(1.0, float(step) / decay_steps)
    return start_eps + fraction * (end_eps - start_eps)

def _select_action(network: AnyMambaDQNNetwork, state_sequence_tensor: torch.Tensor, epsilon: float, num_actions: int, device: torch.device, use_amp: bool) -> int:
    if random.random() < epsilon:
        return random.randrange(num_actions)
    else:
        network.eval()
        with torch.no_grad():
            state_tensor = state_sequence_tensor.unsqueeze(0).to(device) # (1, S, C, H, W)
            with autocast(device.type, dtype=torch.float16 if use_amp else torch.float32, enabled=use_amp):
                q_values, _ = network(state_tensor)
            action = q_values.max(1)[1].item()
        network.train()
        return action

def _train_step(main_net: 'AnyMambaDQNNetwork', target_net: 'AnyMambaDQNNetwork', optimizer: optim.Optimizer, scaler: GradScaler, batch_data: Tuple[torch.Tensor, ...], gamma: float, device: torch.device, use_amp: bool, max_grad_norm: float) -> Tuple[Optional[float], Optional[torch.Tensor]]:
    try:

        state_seq, action, reward, next_state_seq, done, latent_vector_batch, weights, indices = batch_data

        with torch.no_grad():
            with autocast(device_type=device.type, dtype=torch.float16 if use_amp else torch.float32, enabled=use_amp):
                target_q_next, _ = target_net(next_state_seq)
                max_target_q_next = target_q_next.max(1)[0].unsqueeze(1)
                td_target = reward + gamma * max_target_q_next * (1.0 - done)

        main_net.train()
        with autocast(device_type=device.type, dtype=torch.float16 if use_amp else torch.float32, enabled=use_amp):
            current_q, current_latent_vector = main_net(state_seq)
            current_q_selected = current_q.gather(1, action)
            td_errors = td_target - current_q_selected
            elementwise_loss = F.smooth_l1_loss(current_q_selected, td_target, reduction='none')
            loss = (elementwise_loss * weights).mean()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(main_net.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        return loss.item(), td_errors.detach()

    except Exception as e:
        print(f"Error during training step: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def _save_checkpoint(main_net: 'AnyMambaDQNNetwork', optimizer: optim.Optimizer, scaler: GradScaler, total_steps: int, epsilon: float, episode_num: int, use_amp: bool, path: str):
    print(f"\n[INFO] Saving checkpoint to {path} (Step: {total_steps}, Episode: {episode_num})")
    save_dict = {
        'model_state_dict': main_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_steps': total_steps,
        'epsilon': epsilon,
        'episode_num': episode_num,
        'scaler_state_dict': scaler.state_dict() if use_amp else None
    }
    torch.save(save_dict, path)

def train_loop(env: gym.Env,
               obs_shape: Tuple[int, int, int],
               main_net: 'AnyMambaDQNNetwork',
               target_net: 'AnyMambaDQNNetwork',
               optimizer: torch.optim.Optimizer,
               replay_buffer: 'PrioritizedReplayBufferGPU',
               device: torch.device,
               writer: SummaryWriter,
               total_training_steps: int, batch_size: int, sequence_length: int, gamma: float,
               eps_start: float, eps_end: float, eps_decay_steps: int,
               target_update_freq: int, log_interval: int, save_interval: int,
               model_save_dir: str, min_steps: int, current_max_grad_norm: float,
               resume_checkpoint_path: Optional[str] = None):

    num_actions = env.action_space.n
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)

    total_steps, episode_num, epsilon = 0, 0, eps_start
    current_loss = None
    start_time = time.time()

    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        print(f"\n[INFO] Loading checkpoint from {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        main_net.load_state_dict(checkpoint['model_state_dict'])
        target_net.load_state_dict(main_net.state_dict())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if use_amp and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        total_steps = checkpoint.get('total_steps', 0)
        epsilon = checkpoint.get('epsilon', eps_start)
        episode_num = checkpoint.get('episode_num', 0)
        print(
            f"[INFO] Checkpoint loaded. Resuming from step {total_steps}, episode {episode_num}, epsilon {epsilon:.4f}")
    else:
        if resume_checkpoint_path:
            print(f"[WARN] Checkpoint file not found at {resume_checkpoint_path}. Starting from scratch.")
        else:
            print("[INFO] No checkpoint path set. Starting from scratch.")

    state_seq_tensor = torch.zeros((sequence_length, *obs_shape), dtype=torch.uint8, device="cpu")

    print(f"[DEBUG] state_seq_tensor initialized. Shape: {state_seq_tensor.shape}, Device: {state_seq_tensor.device}")

    print(f"\n--- [INFO] Starting Training (Total steps to run: {total_training_steps}) ---")
    while total_steps < total_training_steps:
        episode_num += 1
        episode_reward = 0
        episode_steps = 0

        observation, info = env.reset()
        permuted_obs = np.transpose(np.array(observation, dtype=np.uint8), (2, 0, 1))
        current_observation_tensor = torch.from_numpy(permuted_obs) # (1, 84, 84) 텐서

        for i in range(sequence_length):
            state_seq_tensor[i] = current_observation_tensor

        terminated, truncated = False, False
        while not terminated and not truncated:
            if total_steps >= total_training_steps: break

            total_steps += 1
            episode_steps += 1

            current_state_sequence_for_action = state_seq_tensor.to(device)
            action = _select_action(main_net, current_state_sequence_for_action, epsilon, num_actions, device, use_amp)


            next_observation, reward, terminated, truncated, info = env.step(action)
            permuted_next_obs = np.transpose(np.array(next_observation, dtype=np.uint8), (2, 0, 1))
            episode_reward += reward

            next_state_seq_tensor = torch.roll(state_seq_tensor, shifts=-1, dims=0)
            next_state_seq_tensor[-1] = torch.from_numpy(permuted_next_obs)

            with torch.no_grad():
                _, actual_latent_tensor = main_net(current_state_sequence_for_action.unsqueeze(0))

            done = terminated or truncated
            replay_buffer.push(state_seq_tensor.numpy(), action, reward, next_state_seq_tensor.numpy(), done, actual_latent_tensor)

            state_seq_tensor = next_state_seq_tensor

            can_train = (len(replay_buffer) >= min_steps)
            if can_train:
                batch_data = replay_buffer.sample(total_steps)
                if batch_data:
                    loss_value, td_errors = _train_step(main_net, target_net, optimizer, scaler, batch_data, gamma,
                                                        device, use_amp, current_max_grad_norm)
                    if loss_value is not None and td_errors is not None:
                        current_loss = loss_value
                        replay_buffer.update_priorities(batch_data[-1], td_errors)

            if total_steps % target_update_freq == 0:
                print(f"\n[INFO] Updating target network at step {total_steps}")
                target_net.load_state_dict(main_net.state_dict())

            epsilon = linear_epsilon_decay(total_steps, eps_start, eps_end, eps_decay_steps)

            if total_steps > 0 and total_steps % log_interval == 0:
                elapsed_time = time.time() - start_time
                if not can_train:
                    loss_str = "N/A (Waiting to train)"
                elif current_loss is None:
                    loss_str = "N/A (Train err/No batch/Prev)"
                else:
                    loss_str = f"{current_loss:.4f}"

                BETA_START = 0.4 # 이 변수가 외부에 정의되어 있다고 가정
                current_beta = replay_buffer.beta if hasattr(replay_buffer, 'beta') else BETA_START

                print(f"[LOG] Steps: {total_steps}/{total_training_steps} | Time: {elapsed_time:.2f}s | "
                      f"Episode: {episode_num} | Ep Steps: {episode_steps} | "
                      f"Epsilon: {epsilon:.4f} | Beta: {current_beta:.4f} | Loss: {loss_str} | Buffer: {len(replay_buffer)}")

                if current_loss is not None and can_train:
                    writer.add_scalar("Loss/TD_Loss", current_loss, total_steps)
                writer.add_scalar("Policy/Epsilon", epsilon, total_steps)
                writer.add_scalar("Parameters/Beta", current_beta, total_steps)
                writer.add_scalar("Buffer/Size", len(replay_buffer), total_steps)
                if device.type == 'cuda':
                    try:
                        writer.add_scalar("Memory/GPU_Allocated (MB)", torch.cuda.memory_allocated(device) / 1e6,
                                          total_steps)
                        writer.add_scalar("Memory/GPU_Reserved (MB)", torch.cuda.memory_reserved(device) / 1e6,
                                          total_steps)
                        if use_amp and scaler.get_scale() != 1.0:
                            writer.add_scalar("AMP/Scaler_Scale", scaler.get_scale(), total_steps)
                    except Exception as mem_e:
                        print(f"[WARN] Could not log GPU memory usage - {mem_e}")

                if can_train:
                    current_loss = None

            if total_steps > 0 and total_steps % save_interval == 0:
                safe_env_name = ENV_NAME.replace('/', '_')
                save_path = os.path.join(model_save_dir, f"mlp_mamba_dqn_{safe_env_name}_steps_{total_steps}.pth")
                _save_checkpoint(main_net, optimizer, scaler, total_steps, epsilon, episode_num, use_amp, save_path)
            if total_steps >= total_training_steps:
                break

        if episode_steps > 0:
            print(
                f"[INFO] Episode {episode_num} finished after {episode_steps} steps. Total reward: {episode_reward:.2f}")
            writer.add_scalar("Reward/Episode_TotalReward", episode_reward,
                              total_steps)
            writer.add_scalar("Steps/Episode_Length", episode_steps,
                              total_steps)
            writer.add_scalar("Reward/Episode_TotalReward_vs_EpisodeNum",
                              episode_reward, episode_num)
            writer.add_scalar("Steps/Episode_Length_vs_EpisodeNum", episode_steps,
                              episode_num)
        else:
            print(f"[WARN] Episode {episode_num} finished with 0 steps. This might indicate an issue.")

    if total_steps > 0:
        safe_env_name = ENV_NAME.replace('/', '_')
        final_save_path = os.path.join(model_save_dir, f"mlp_mamba_dqn_{safe_env_name}_final_steps_{total_steps}.pth")
        _save_checkpoint(main_net, optimizer, scaler, total_steps, epsilon, episode_num, use_amp, final_save_path)

    print(f"\n--- [INFO] Training Finished (Total steps: {total_steps}) ---")


if __name__ == "__main__":
    print(f"[Main] 스크립트 실행 시작 (MambaDQN - Model: {MODEL_CLASS_NAME})")
    set_seed(SEED)
    device = get_device()

    print(f"[Main] 환경 '{ENV_NAME}' 생성 중...")
    env = create_env(ENV_NAME)
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    print(f"[Main] 환경 생성 완료. Obs Shape: {obs_shape}, Actions: {num_actions}")

    pytorch_input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
    print(f"[Main] PyTorch Input Shape로 변환: {pytorch_input_shape}")

    current_max_grad_norm = GRAD_CLIP_CONFIG.get(MODEL_CLASS_NAME, GRAD_CLIP_CONFIG["default"])
    print(f"[Main] Gradient Clipping max_norm: {current_max_grad_norm}")

##network##
    D_MODEL, D_STATE, D_CONV, EXPAND = 128, 32, 4, 2
    FC_HIDDEN_DIM1 = 128
    FC_HIDDEN_DIM2 = 128

    print(
        f"[Main] 네트워크 초기화 중... PyTorch Input Shape: {pytorch_input_shape}, Seq_len: {SEQUENCE_LENGTH}, Actions: {num_actions}")
    main_net = SELECTED_MODEL_CLASS(
        input_shape=pytorch_input_shape,
        sequence_length=SEQUENCE_LENGTH,
        num_actions=num_actions,
        d_model=D_MODEL,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
        fc_hidden_dim1=FC_HIDDEN_DIM1,
        fc_hidden_dim2=FC_HIDDEN_DIM2
    ).to(device)
    target_net = SELECTED_MODEL_CLASS(
        input_shape=pytorch_input_shape,
        sequence_length=SEQUENCE_LENGTH,
        num_actions=num_actions,
        d_model=D_MODEL,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
        fc_hidden_dim1=FC_HIDDEN_DIM1,
        fc_hidden_dim2=FC_HIDDEN_DIM2
    ).to(device)
    target_net.load_state_dict(main_net.state_dict())
    target_net.eval()
    print("[Main] 네트워크 초기화 완료.")

    optimizer = torch.optim.AdamW(main_net.parameters(), lr=LEARNING_RATE,amsgrad=True)
    print("[Main] 옵티마이저 초기화 완료.")

    print(f"[Main] 리플레이 버퍼 초기화 중... PyTorch Input Shape: {pytorch_input_shape}, Latent_dim: {D_MODEL}")
    replay_buffer = PrioritizedReplayBufferGPU(
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        input_shape=pytorch_input_shape,
        latent_dim=D_MODEL,
        device=device,
        alpha=ALPHA,
        beta_start=BETA_START,
        beta_frames=TOTAL_TRAINING_STEPS,
    )
    print(f"[Main] 리플레이 버퍼 초기화 완료. Buffer capacity: {BUFFER_SIZE}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    current_time_str = time.strftime("%Y%m%d-%H%M%S")
    log_dir_base = "logs"
    log_dir_name = f"{MODEL_CLASS_NAME.lower()}_{ENV_NAME.replace('/', '_')}_{current_time_str}"
    log_dir = os.path.join(log_dir_base, log_dir_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[Main] Model save directory: {os.path.abspath(MODEL_SAVE_DIR)}")
    print(f"[Main] TensorBoard log directory: {os.path.abspath(log_dir)}")
    resume_from = None

    print("[Main] train_loop 함수 호출 준비...")
    try:
        train_loop(
            env=env,
            obs_shape=pytorch_input_shape, # ★★★ obs_shape 대신 pytorch_input_shape 사용
            main_net=main_net,
            target_net=target_net,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            device=device,
            writer=writer,
            total_training_steps=TOTAL_TRAINING_STEPS,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            gamma=GAMMA,
            eps_start=EPS_START,
            eps_end=EPS_END,
            eps_decay_steps=EPS_DECAY_STEPS,
            target_update_freq=TARGET_UPDATE_FREQ,
            log_interval=LOG_INTERVAL,
            save_interval=SAVE_INTERVAL,
            model_save_dir=MODEL_SAVE_DIR,
            min_steps=MIN_STEPS_FOR_TRAINING,
            current_max_grad_norm=current_max_grad_norm,
            resume_checkpoint_path=resume_from
        )

    except KeyboardInterrupt:
        print("\n[Main] 사용자에 의해 학습이 중단되었습니다.")
    except Exception as e:
        print(f"[Main_ERROR] 메인 루프에서 예기치 않은 오류 발생: {e}")

    finally:
        print("\n[Main] 학습 루프 종료. 자원 해제 시도...")
        if 'writer' in locals() and writer is not None:
            writer.close()
            print("[Main] TensorBoard writer가 닫혔습니다.")
        if 'env' in locals() and env is not None:
            env.close()
            print("[Main] 환경이 닫혔습니다.")
        print("[Main] 스크립트 실행 종료.")
