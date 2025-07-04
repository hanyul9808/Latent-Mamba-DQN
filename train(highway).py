import torch
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import highway_env # noqa: F401
import numpy as np
import time
import random
import os
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from typing import Tuple, Optional, Union
from network.mamba_dqn_network import latentMambaDQNNetwork
from replay_buffer.per_buffer import PrioritizedReplayBufferGPU


SELECTED_MODEL_CLASS = latentMambaDQNNetwork

MODEL_CLASS_NAME = SELECTED_MODEL_CLASS.__name__
AnyMambaDQNNetwork = Union[latentMambaDQNNetwork]


ENV_NAME = 'highway-fast-v0'
INPUT_DIM = None
SEQUENCE_LENGTH = 8
MLP_HIDDEN_DIM = 256
D_MODEL = 128
D_STATE = 32
D_CONV = 4
EXPAND = 2

GRAD_CLIP_CONFIG = {
    "latentMambaDQNNetwork": 0.1,       # Nomal MLP 모델의 max_norm 값
    "default": 1.0                     # 혹시 정의되지 않은 모델일 경우를 위한 기본값
}

BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.8
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_STEPS = 14000
TARGET_UPDATE_FREQ = 50
LEARNING_RATE = 5e-4
TOTAL_TRAINING_STEPS = 20000
ALPHA = 0.6
BETA_START = 0.4

LOG_INTERVAL = 50
SAVE_INTERVAL = 10000

MODEL_SAVE_DIR = f"saved_models_latent_{MODEL_CLASS_NAME.lower()}"
SEED = 42
MIN_STEPS_FOR_TRAINING = 200

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

def create_env(env_name_str: str) -> Tuple[gym.Env, int]:
    env = gym.make(env_name_str, render_mode=None)
    if isinstance(env.observation_space, gym.spaces.Dict):
        obs_shape = env.observation_space['observation'].shape
    else:
        obs_shape = env.observation_space.shape
    input_dim_val = int(np.prod(obs_shape))
    return env, input_dim_val


def linear_epsilon_decay(step: int, start_eps: float, end_eps: float, decay_steps: int) -> float:
    fraction = min(1.0, float(step) / decay_steps)
    return start_eps + fraction * (end_eps - start_eps)

def _select_action(network: AnyMambaDQNNetwork,
                   state_sequence_tensor: torch.Tensor,
                   epsilon: float,
                   num_actions: int,
                   device: torch.device,
                   use_amp: bool) -> int:
    if random.random() < epsilon:
        return random.randrange(num_actions)

    else:
        network.eval()
        with torch.no_grad():
            state_tensor = state_sequence_tensor.unsqueeze(0) # (1, S, input_dim)
            with autocast(device.type, dtype=torch.float16 if use_amp else torch.float32, enabled=use_amp):
                q_values, _ = network(state_tensor)
            action = q_values.max(1)[1].item()
        network.train()
        return action

def _train_step(main_net: 'AnyMambaDQNNetwork',
                target_net: 'AnyMambaDQNNetwork',
                optimizer: optim.Optimizer,
                scaler: GradScaler,
                batch_data: Tuple[torch.Tensor, ...],
                gamma: float,
                device: torch.device,
                use_amp: bool,
                max_grad_norm: float) -> Tuple[Optional[float], Optional[torch.Tensor]]:

    try:
        state_seq, action, reward, next_state_seq, done, latent_vector_batch, weights, indices = batch_data

        with torch.no_grad():
            with autocast(device_type=device.type, dtype=torch.float16 if use_amp else torch.float32, enabled=use_amp):
                target_q_next, _ = target_net(next_state_seq)
                max_target_q_next = target_q_next.max(1)[0].unsqueeze(1)
                td_target = reward + gamma * max_target_q_next * (1.0 - done.float())

        main_net.train()
        with autocast(device_type=device.type, dtype=torch.float16 if use_amp else torch.float32, enabled=use_amp):
            current_q, current_latent_vector = main_net(state_seq)
            current_q_selected = current_q.gather(1, action)
            td_errors = td_target.float() - current_q_selected.float()
            elementwise_loss = F.smooth_l1_loss(current_q_selected.float(), td_target.float(), reduction='none')
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

    except Exception as e:
        print(f"Error during training step: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def _save_checkpoint(main_net: 'AnyMambaDQNNetwork',
                     optimizer: optim.Optimizer,
                     scaler: GradScaler,
                     total_steps: int,
                     epsilon: float,
                     episode_num: int,
                     use_amp: bool,
                     path: str):
    print(f"\n[INFO] Saving checkpoint to {path} (Step: {total_steps}, Episode: {episode_num})")
    save_dict = {
        'model_state_dict': main_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_steps': total_steps,
        'epsilon': epsilon,
        'episode_num': episode_num
    }

    if use_amp:
        save_dict['scaler_state_dict'] = scaler.state_dict()
    torch.save(save_dict, path)


def train_loop(env: gym.Env,
               input_dim: int,
               main_net: 'AnyMambaDQNNetwork',
               target_net: 'AnyMambaDQNNetwork',
               optimizer: optim.Optimizer,
               replay_buffer: 'PrioritizedReplayBufferGPU',
               device: torch.device,
               writer: SummaryWriter,
               total_training_steps: int,
               batch_size: int,
               sequence_length: int,
               gamma: float,
               eps_start: float,
               eps_end: float,
               eps_decay_steps: int,
               target_update_freq: int,
               log_interval: int,
               save_interval: int,
               model_save_dir: str,
               min_buffer_size: int,
               min_steps: int,
               current_max_grad_norm: float,
               resume_checkpoint_path: Optional[str] = None):

    global ENV_NAME
    global BETA_START

    num_actions = env.action_space.n
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)

    total_steps = 0
    episode_num = 0
    epsilon = eps_start
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

    state_seq_tensor = torch.zeros((sequence_length, input_dim), dtype=torch.float32, device=device)
    print(f"[DEBUG] state_seq_tensor initialized. Shape: {state_seq_tensor.shape}, Device: {state_seq_tensor.device}")

    print(f"\n--- [INFO] Starting Training (Total steps to run: {total_training_steps}) ---")
    while total_steps < total_training_steps:
        episode_num += 1
        episode_reward = 0
        episode_steps = 0

        observation, info = env.reset()
        flattened_observation = observation.reshape(-1).astype(np.float32)
        if flattened_observation.shape[0] != input_dim:
            print(
                f"[WARN] Observation dimension mismatch after flatten! Expected {input_dim}, Got {flattened_observation.shape[0]}. Check env config or input_dim.")
        current_observation_tensor = torch.from_numpy(flattened_observation).to(device=device)

        for i in range(sequence_length):
            state_seq_tensor[i] = current_observation_tensor

        terminated = False
        truncated = False

        while not terminated and not truncated:
            if total_steps >= total_training_steps:
                break

            total_steps += 1
            episode_steps += 1

            current_state_sequence_tensor_for_action = state_seq_tensor.clone()

            with torch.no_grad():
                main_net.eval()
                _, actual_latent_tensor = main_net(current_state_sequence_tensor_for_action.unsqueeze(0))
                main_net.train()

            action = _select_action(main_net, current_state_sequence_tensor_for_action, epsilon,
                                    num_actions, device, use_amp)

            next_observation, reward, terminated, truncated, info = env.step(action)
            flattened_next_observation = next_observation.reshape(-1).astype(np.float32)
            if flattened_next_observation.shape[0] != input_dim:
                print(
                    f"[WARN] Next observation dimension mismatch after flatten! Expected {input_dim}, Got {flattened_next_observation.shape[0]}.")
            next_observation_tensor = torch.from_numpy(flattened_next_observation).to(device=device)
            episode_reward += reward

            state_seq_tensor = torch.roll(state_seq_tensor, shifts=-1, dims=0)
            state_seq_tensor[-1] = next_observation_tensor

            next_state_sequence_tensor_for_buffer = state_seq_tensor.clone()
            done = terminated or truncated

            replay_buffer.push(current_state_sequence_tensor_for_action,
                               action,
                               reward,
                               next_state_sequence_tensor_for_buffer,
                               done,
                               actual_latent_tensor)

            can_train = (replay_buffer.size >= BATCH_SIZE and total_steps >= min_steps)

            if can_train:
                batch_data = replay_buffer.sample(total_steps)
                if batch_data is not None:
                    loss_value, td_errors = _train_step(main_net, target_net, optimizer, scaler,
                                                        batch_data, gamma, device, use_amp,max_grad_norm=current_max_grad_norm)
                    if loss_value is not None and td_errors is not None:
                        current_loss = loss_value
                        leaf_indices = batch_data[-1]
                        replay_buffer.update_priorities(leaf_indices, td_errors)
                    else:
                        if total_steps % (log_interval // 2 if log_interval > 1 else 1) == 0:
                            print(
                                f"[WARN] Training step at total_steps={total_steps} failed or returned None. Loss not updated.")

            if total_steps > 0 and total_steps % target_update_freq == 0:
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

                current_beta = replay_buffer.beta if hasattr(replay_buffer, 'beta') and isinstance(replay_buffer, PrioritizedReplayBufferGPU) else BETA_START

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
                save_path = os.path.join(model_save_dir, f"mlp_mamba_dqn_{ENV_NAME}_steps_{total_steps}.pth")
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
        final_save_path = os.path.join(model_save_dir, f"mlp_mamba_dqn_{ENV_NAME}_final_steps_{total_steps}.pth")
        _save_checkpoint(main_net, optimizer, scaler, total_steps, epsilon, episode_num, use_amp, final_save_path)

    print(f"\n--- [INFO] Training Finished (Total steps: {total_steps}) ---")


if __name__ == "__main__":
    print(f"[Main] 스크립트 실행 시작 (MambaDQN - Model: {MODEL_CLASS_NAME})")
    set_seed(SEED)
    device = get_device()

    print(f"[Main] 환경 '{ENV_NAME}' 생성 및 input_dim 결정 중...")
    env, local_input_dim = create_env(ENV_NAME)
    num_actions = env.action_space.n
    INPUT_DIM = local_input_dim # 전역 INPUT_DIM 업데이트
    print(f"[Main] 환경 생성 완료. Determined input_dim: {INPUT_DIM}, Actions: {num_actions}")

    current_max_grad_norm = GRAD_CLIP_CONFIG.get(MODEL_CLASS_NAME, GRAD_CLIP_CONFIG["default"])
    print(f"[Main] Using Gradient Clipping max_norm: {current_max_grad_norm} for model '{MODEL_CLASS_NAME}'")


    print(f"[Main] 네트워크 초기화 중... Input_dim: {INPUT_DIM}, Seq_len: {SEQUENCE_LENGTH}, Actions: {num_actions}, D_Model: {D_MODEL}")
    main_net = SELECTED_MODEL_CLASS(
        input_dim=INPUT_DIM,
        sequence_length=SEQUENCE_LENGTH,
        num_actions=num_actions,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        d_model=D_MODEL,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND
    ).to(device)
    target_net = SELECTED_MODEL_CLASS(
        input_dim=INPUT_DIM,
        sequence_length=SEQUENCE_LENGTH,
        num_actions=num_actions,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        d_model=D_MODEL,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND
    ).to(device)
    target_net.load_state_dict(main_net.state_dict())
    target_net.eval()
    print("[Main] 네트워크 초기화 완료.")

    print("[Main] 옵티마이저 초기화 중...")
    optimizer = optim.AdamW(main_net.parameters(), lr=LEARNING_RATE)
    print("[Main] 옵티마이저 초기화 완료.")

    print(f"[Main] 리플레이 버퍼 초기화 중... Input_dim: {INPUT_DIM}, Seq_len: {SEQUENCE_LENGTH}, Latent_dim: {D_MODEL}")
    replay_buffer = PrioritizedReplayBufferGPU(
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        input_dim=INPUT_DIM,
        latent_dim=D_MODEL,
        device=device,
        alpha=ALPHA,
        beta_start=BETA_START,
        beta_frames=TOTAL_TRAINING_STEPS,
        state_dtype=torch.float32
    )
    print(f"[Main] 리플레이 버퍼 초기화 완료. Buffer size: {BUFFER_SIZE}, Batch size: {BATCH_SIZE}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    current_time_str = time.strftime("%Y%m%d-%H%M%S")
    log_dir_base = "logs"
    log_dir_name = f"{MODEL_CLASS_NAME.lower()}_{ENV_NAME}_{current_time_str}"
    log_dir = os.path.join(log_dir_base, log_dir_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[Main] Model save directory: {os.path.abspath(MODEL_SAVE_DIR)}")
    print(f"[Main] TensorBoard log directory: {os.path.abspath(log_dir)}")
    print(f"         To view TensorBoard, run: tensorboard --logdir=\"{os.path.abspath(log_dir_base)}\"")

    resume_from = None
    if resume_from and os.path.exists(resume_from):
        print(f"[Main] 체크포인트에서 이어 시작 예정: {resume_from}")
    else:
        if resume_from:
            print(f"[Main] 체크포인트 파일 없음: {resume_from}. 처음부터 학습 시작.")
            resume_from = None
        else:
            print("[Main] 처음부터 학습 시작 예정 (체크포인트 설정 없음).")

    print("[Main] train_loop 함수 호출 준비...")
    try:
        train_loop(
            env=env,
            input_dim=INPUT_DIM,
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
            min_buffer_size=BATCH_SIZE,
            min_steps=MIN_STEPS_FOR_TRAINING,
            resume_checkpoint_path=resume_from,
            current_max_grad_norm=current_max_grad_norm
        )

    except KeyboardInterrupt:
        print("\n[Main] 사용자에 의해 학습이 중단되었습니다.")
    except Exception as e:
        print(f"[Main_ERROR] 메인 루프에서 예기치 않은 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Main] 학습 루프 종료. 자원 해제 시도...")
        if 'writer' in locals() and writer is not None:
            writer.close()
            print("[Main] TensorBoard writer가 닫혔습니다.")

        if 'env' in locals() and env is not None:
            env.close()
            print("[Main] 환경이 닫혔습니다.")
        if 'MODEL_CLASS_NAME' in locals() and 'current_max_grad_norm' in locals():
            print(f"[Main_Summary] 최종 사용 네트워크: {MODEL_CLASS_NAME}, 적용된 클리핑 계수(max_norm): {current_max_grad_norm}")
        else:
            print("[Main_Summary] 네트워크 또는 클리핑 계수 정보를 로드하지 못했습니다.")

        print("[Main] 스크립트 실행 종료.")
