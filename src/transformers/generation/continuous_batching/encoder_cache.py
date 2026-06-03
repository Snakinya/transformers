# Copyright 2026 The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections import deque
from itertools import repeat

import torch

from ...configuration_utils import PretrainedConfig
from ...generation.configuration_utils import ContinuousBatchingConfig
from .requests import RequestState


# TODO: add block-based indexing
class EncoderCache:

    cache: torch.Tensor
    REQUEST_ID_KEY: str = "_cb_request_id"

    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_tokens: int,
        use_async_batching: bool,
        model_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.use_async_batching = use_async_batching
        self.image_token_id = config.image_token_id
        if not isinstance(self.image_token_id, int) or self.image_token_id <= 0:
            raise ValueError(f"Image token ID must be a positive integer but got {self.image_token_id = }")
        # Create the actual cache tensor
        cache_size = max(16384, max_batch_tokens)
        cache_shape = (cache_size, config.text_config.hidden_size)
        self.cache = torch.empty(cache_shape, dtype=model_dtype, device=device)
        # Create bookkeeping data structures
        self.free_blocks = deque(range(cache_size))
        self.allocated_blocks_masks: dict[str, torch.Tensor] = {}
        self.embeddings_lengths: dict[str, int] = {}
        self.outgoing_requests: list[str] = [] # TODO: BUG: this is not used, it needs to be called when the batch is done

    def can_store_mm_embeddings(self, state: RequestState) -> bool:
        """Checks if there is enough space in the encoder cache to store the multimodal embeddings."""
        # Retrieve the number of multimodal embeddings from the multimodal data (or compute and cache it)
        num_mm_embeddings = self.embeddings_lengths.get(state.request_id)
        if num_mm_embeddings is None:
            input_ids = torch.tensor(state.initial_tokens, device="cpu", dtype=torch.int32)
            num_mm_embeddings = (input_ids == self.image_token_id).sum().item()
            self.embeddings_lengths[state.request_id] = num_mm_embeddings
        return len(self.free_blocks) >= num_mm_embeddings

    def allocate_blocks(self, state: RequestState) -> None:
        """Allocates blocks for a request. This should only be called once per request."""
        # Get the list of allocated blocks for the request
        num_mm_embeddings = self.embeddings_lengths.pop(state.request_id)  # this value will never be used again
        allocated_blocks = [self.free_blocks.popleft() for _ in range(num_mm_embeddings)]
        # Infer the allocated blocks mask
        input_ids = torch.tensor(state.initial_tokens, device="cpu", dtype=torch.int32)
        img_mask = (input_ids == self.image_token_id)
        input_ids.fill_(-1)
        input_ids[img_mask] = torch.tensor(allocated_blocks, device="cpu", dtype=torch.int32)
        self.allocated_blocks_masks[state.request_id] = input_ids
        # TODO: this could be optimized by truncating from the first and last img tokens

    def extend_read_indices(
        self, request_id: str, past_length: int, query_length: int, read_indices: list[int]
    ) -> bool:
        """
        Extends the list of indices being read from the encoder cache for a given request. Returns True if any
        multimodal embeddings are read in this batch, False otherwise. For instance, if the inital tokens and allocated
        blocks are as follows:

            Initial tokens:   [xxx, xxx, xxx, img, img, img, xxx]
            Allocated blocks: [ -1,  -1,  -1,   0,   1,   3,  -1]
            (index)              0    1    2    3    4    5    6

        Then for a past length of 3 and a query length of 5, the read indices will be:

            Read indices:     [                 0,   1,   3,  -1,  -1]

        and the function will return True because there are actual cache reads (block 0, 1 and 3 are read).
        """
        block_table = self.allocated_blocks_masks.get(request_id)
        # Only compute read indices if the request has allocated blocks
        if block_table is not None:
            intersection = block_table[past_length:past_length + query_length].tolist()
            missing_indices = query_length - len(intersection)
            # Check if any of the multimodal embeddings for this request are read in this batch
            cache_read = (block_table[past_length:past_length + query_length] != -1).any().item()
            # Check if all the multimodal embeddings for this request have been read
            if past_length + query_length >= len(block_table):
                self.outgoing_requests.append(request_id)
        else:
            intersection = []
            missing_indices = query_length
            cache_read = False
        # Extend the read indices
        read_indices.extend(intersection)
        read_indices.extend(repeat(-1, missing_indices))
        return cache_read

    def store_mm_embeddings(self, request_id: str, image_features: torch.Tensor) -> None:
        """Stores the multimodal embeddings for a request in the encoder cache."""
        # Retrieve the allocated blocks mask for the request
        allocated_blocks_mask = self.allocated_blocks_masks.get(request_id)
        if allocated_blocks_mask is None:
            raise ValueError(f"Request {request_id} has no allocated blocks mask")
        # Extract the allocated blocks from the mask
        mask = allocated_blocks_mask != -1
        allocated_blocks = allocated_blocks_mask[mask].to(self.cache.device)
        # Store the multimodal embeddings in the cache
        self.cache[allocated_blocks] = image_features

    def release_outgoing_requests(self) -> None:
        """Releases the outgoing requests from the encoder cache."""
        # Loop until there are no outgoing requests
        while self.outgoing_requests:
            request_id = self.outgoing_requests.pop()
            # Retrieve the list of blocks to free
            allocated_blocks_mask = self.allocated_blocks_masks.pop(request_id, None)
            if allocated_blocks_mask is None and not self.use_async_batching:  # impossible in sync mode
                raise ValueError(f"Cannot release {request_id} because it has no allocated blocks mask")
            mask = allocated_blocks_mask != -1
            blocks_to_free = allocated_blocks_mask[mask].tolist()
            # Actually free the blocks
            self.free_blocks.extend(blocks_to_free)
