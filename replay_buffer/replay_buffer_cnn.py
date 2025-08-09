import torch
import numpy as np

class SumTreeTorchGPU:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = torch.zeros(self.tree_size, dtype=torch.float32, device=device)
        self.data_pointer = 0
        self.device = device
        self.epsilon = 1e-6

    def _propagate(self, tree_idx: int, change: torch.Tensor):
        change = change.to(self.device)
        while tree_idx > 0:
            parent = (tree_idx - 1) // 2
            self.tree[parent] += change
            tree_idx = parent

    def update(self, tree_idx: int, priority: torch.Tensor):
        priority = priority.to(self.device).clamp(min=self.epsilon)
        tree_idx = int(tree_idx)
        if not (self.capacity - 1 <= tree_idx < self.tree_size):
            print(f"Warning (SumTree.update): Invalid tree_idx {tree_idx} provided.")
            return

        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(self, priority: torch.Tensor):
        priority = priority.to(self.device).clamp(min=self.epsilon)
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    @torch.no_grad()
    def _retrieve_iterative(self, s: torch.Tensor) -> int:
        s = s.to(self.device)
        idx = 0
        while True:
            left_child_idx = 2 * idx + 1
            if left_child_idx >= self.tree_size:
                break
            if s <= self.tree[left_child_idx]:
                idx = left_child_idx
            else:
                s = s - self.tree[left_child_idx]
                idx = left_child_idx + 1
        return int(idx)

    @torch.no_grad()
    def get_leaf(self, s: torch.Tensor) -> tuple[int, torch.Tensor, int]:
        leaf_idx = self._retrieve_iterative(s)
        data_idx = leaf_idx - self.capacity + 1
        if not (0 <= data_idx < self.capacity):
            print(
                f"Critical Warning (SumTree.get_leaf): Invalid data_idx {data_idx} derived from leaf_idx {leaf_idx}. Check SumTree logic.")
            data_idx = data_idx % self.capacity
        return leaf_idx, self.tree[leaf_idx], data_idx

    def total_priority(self) -> torch.Tensor:
        return self.tree[0]

    @torch.no_grad()
    def get_max_leaf_priority(self) -> torch.Tensor:
        leaf_start = self.capacity - 1
        leaf_values = self.tree[leaf_start: self.tree_size]
        if leaf_values.numel() == 0:
            return torch.tensor(1.0, device=self.device, dtype=torch.float32)
        max_priority = torch.max(leaf_values)
        one_tensor = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        return max_priority if max_priority > self.epsilon else one_tensor

from typing import Tuple
class PrioritizedReplayBufferGPU:
    def __init__(self, buffer_size: int, batch_size: int, sequence_length: int,
                 input_shape: Tuple[int, int, int],
                 latent_dim: int,
                 device: torch.device,
                 alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000):


        self.capacity = buffer_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.device = device

        buffer_shape = (buffer_size, sequence_length, *self.input_shape)
        self.states = torch.zeros(buffer_shape, dtype=torch.uint8, device=device)
        self.next_states = torch.zeros(buffer_shape, dtype=torch.uint8, device=device)

        self.actions = torch.zeros((buffer_size, 1), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.bool, device=device)
        self.latent_vectors = torch.zeros((buffer_size, latent_dim), dtype=torch.float32, device=device)

        self.sum_tree = SumTreeTorchGPU(buffer_size, device)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.epsilon = 1e-6

        self.position = 0
        self.size = 0

    def push(self, state_sequence: np.ndarray, action: int, reward: float, next_state_sequence: np.ndarray, done: bool,
             latent_vector: torch.Tensor):

        max_priority = self.sum_tree.get_max_leaf_priority()

        current_pos = self.position

        self.states[current_pos] = torch.from_numpy(state_sequence).to(self.device)
        self.next_states[current_pos] = torch.from_numpy(next_state_sequence).to(self.device)

        self.actions[current_pos, 0] = action
        self.rewards[current_pos, 0] = reward
        self.dones[current_pos, 0] = done
        self.latent_vectors[current_pos] = latent_vector.squeeze(0).detach()

        self.sum_tree.add(max_priority)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    @torch.no_grad()
    def sample(self, current_step: int) -> tuple | None:
        if self.size < self.batch_size:
            return None

        indices = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        priorities = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        leaf_indices_in_tree = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)

        total_p = self.sum_tree.total_priority()
        if total_p < 1e-9:
            print("Warning: Total priority is near zero. Cannot perform prioritized sampling.")
            return None

        segment = total_p / self.batch_size
        s_values = torch.rand(self.batch_size, device=self.device) * segment + \
                   torch.arange(self.batch_size, device=self.device) * segment

        for i in range(self.batch_size):
            leaf_idx, priority, data_idx = self.sum_tree.get_leaf(s_values[i])
            indices[i] = data_idx
            priorities[i] = priority
            leaf_indices_in_tree[i] = leaf_idx

        sampling_probabilities = priorities / total_p
        self.beta = min(1.0, self.beta_start + self.beta_increment * current_step)
        weights = (self.size * sampling_probabilities + 1e-9) ** (-self.beta)

        if weights.max() > 1e-9:
            weights = weights / weights.max()
        else:
            weights.fill_(1.0)

        weights_batch = weights.unsqueeze(1)

        state_batch = self.states[indices]
        action_batch = self.actions[indices]
        reward_batch = self.rewards[indices]
        next_state_batch = self.next_states[indices]
        done_batch = self.dones[indices].float()
        latent_vector_batch = self.latent_vectors[indices]

        return (state_batch, action_batch, reward_batch, next_state_batch,
                done_batch, latent_vector_batch, weights_batch,
                leaf_indices_in_tree)

    @torch.no_grad()
    def update_priorities(self, leaf_indices: torch.Tensor, td_errors: torch.Tensor):
        if td_errors.ndim > 1:
            td_errors = td_errors.squeeze()

        priorities = (td_errors.abs() + self.epsilon) ** self.alpha

        if not torch.all(torch.isfinite(priorities)):
            print(f"Warning: Detected non-finite values in calculated priorities: {priorities}")
            priorities = torch.nan_to_num(priorities, nan=self.epsilon, posinf=1e6, neginf=self.epsilon)
            priorities.clamp_(min=self.epsilon)

        for i in range(leaf_indices.shape[0]):
            leaf_idx = leaf_indices[i].item()
            priority = priorities[i]
            self.sum_tree.update(leaf_idx, priority)

    def __len__(self):
        return self.size
