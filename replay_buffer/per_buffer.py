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

class PrioritizedReplayBufferGPU:

    def __init__(self, buffer_size: int, batch_size: int, sequence_length: int,
                 input_dim: int,
                 latent_dim: int,
                 device: torch.device,
                 alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000,
                 gamma: float = 0.99,
                 state_dtype: torch.dtype = torch.float32
                 ):

        self.capacity = buffer_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        self.state_dtype = state_dtype

        buffer_shape = (buffer_size, sequence_length, input_dim)
        self.states = torch.zeros(buffer_shape, dtype=state_dtype, device=device)
        self.actions = torch.zeros((buffer_size, 1), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros(buffer_shape, dtype=state_dtype, device=device)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.latent_vectors = torch.zeros((buffer_size, latent_dim), dtype=torch.float32, device=device)

        self.sum_tree = SumTreeTorchGPU(buffer_size, device)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.epsilon = 1e-6

        self.position = 0
        self.size = 0
        self.gamma = gamma

        print(f"PrioritizedReplayBufferGPU (Vector Input & Latent Storage) initialized.")
        print(f"  Device: {self.device}. Buffer Size: {buffer_size}. Batch Size: {batch_size}.")
        print(
            f"  Sequence Length: {sequence_length}. Input Dim: {input_dim}. Latent Dim: {latent_dim}. State Dtype: {self.state_dtype}")

    def push(self, state_sequence, action: int, reward: float, next_state_sequence, done: bool,
             latent_vector: torch.Tensor):
        def prepare_sequence(seq_data):
            if not isinstance(seq_data, torch.Tensor):
                if isinstance(seq_data, np.ndarray):
                    seq_data = torch.from_numpy(seq_data)
                else:
                    try:
                        seq_data = torch.tensor(seq_data)
                    except Exception as e:
                        raise TypeError(
                            f"state_sequence must be provided as torch.Tensor or compatible type (e.g., numpy array). Error: {e}")
            seq_tensor = seq_data.to(device=self.device, dtype=self.state_dtype)
            expected_shape = (self.sequence_length, self.input_dim)
            if seq_tensor.shape != expected_shape:
                raise ValueError(f"Sequence shape mismatch. Expected: {expected_shape}, Got: {seq_tensor.shape}")
            return seq_tensor

        try:
            state_seq_tensor = prepare_sequence(state_sequence)
            next_state_seq_tensor = prepare_sequence(next_state_sequence)
        except (TypeError, ValueError) as e:
            print(f"Error processing sequences in push method: {e}")
            return

        action_tensor = torch.tensor([[action]], dtype=torch.int64, device=self.device)
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor([[float(done)]], dtype=torch.float32, device=self.device)

        if latent_vector.shape != (1, self.latent_dim) and latent_vector.shape != (self.latent_dim,):
            raise ValueError(
                f"Latent vector shape mismatch. Expected (1, {self.latent_dim}) or ({self.latent_dim},), Got: {latent_vector.shape}")

        processed_latent_vector = latent_vector.squeeze(0).detach().to(device=self.device, dtype=torch.float32)
        if processed_latent_vector.shape != (self.latent_dim,):
            processed_latent_vector = latent_vector.detach().to(device=self.device,
                                                                dtype=torch.float32)
            if processed_latent_vector.shape != (self.latent_dim,):
                raise ValueError(
                    f"Final Latent vector shape mismatch after processing. Expected ({self.latent_dim},), Got: {processed_latent_vector.shape}")

        current_pos = self.position
        self.states[current_pos] = state_seq_tensor
        self.actions[current_pos] = action_tensor
        self.rewards[current_pos] = reward_tensor
        self.next_states[current_pos] = next_state_seq_tensor
        self.dones[current_pos] = done_tensor
        self.latent_vectors[current_pos] = processed_latent_vector

        max_priority = self.sum_tree.get_max_leaf_priority()
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
        max_weight = torch.max(weights)
        if max_weight > 1e-9:
            weights = weights / max_weight
        else:
            weights.fill_(1.0)

        weights_batch = weights.unsqueeze(1)
        state_batch = self.states[indices]
        action_batch = self.actions[indices]
        reward_batch = self.rewards[indices]
        next_state_batch = self.next_states[indices]
        done_batch = self.dones[indices]
        latent_vector_batch = self.latent_vectors[indices]

        return (state_batch, action_batch, reward_batch, next_state_batch,
                done_batch, latent_vector_batch, weights_batch,
                leaf_indices_in_tree)

    @torch.no_grad()
    def update_priorities(self, leaf_indices: torch.Tensor, td_errors: torch.Tensor):
        leaf_indices = leaf_indices.to(self.device)
        td_errors = td_errors.to(self.device)
        if td_errors.ndim > 1:
            td_errors = td_errors.squeeze()
        if td_errors.shape[0] != leaf_indices.shape[0]:
            raise ValueError(
                f"Batch size mismatch between leaf_indices ({leaf_indices.shape[0]}) and td_errors ({td_errors.shape[0]})")

        priorities = (torch.abs(td_errors) + self.epsilon) ** self.alpha
        if not torch.all(torch.isfinite(priorities)):
            print(f"Warning: Detected non-finite values in calculated priorities: {priorities}")
            priorities = torch.nan_to_num(priorities, nan=self.epsilon, posinf=1e6, neginf=self.epsilon)
            priorities = torch.clamp(priorities, min=self.epsilon)

        for i in range(leaf_indices.shape[0]):
            leaf_idx = leaf_indices[i].item()
            priority = priorities[i]
            self.sum_tree.update(leaf_idx, priority)

    def __len__(self):
        return self.size
