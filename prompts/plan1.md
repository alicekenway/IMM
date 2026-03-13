  I will tell you my idea and please help me judge whether it can work and is a good idea.
present design is a failure since we can not full no grad the memory filling part, other wise, the memory writing projection will not be trained. 
and I find the memory bank should not be existed in the training stage, it is only useful in the inference stage.
so my plan is, the input include the history and the prompt+present.
for each layer will create 2 based model object, share the same parameter
one for history, go to pool, go to k and v
one for present input and prompt, only the input go to q, the calcuate the output with  q k v, merge it with the hidden, then for next layer.
using 2 nets here is to prevent the present text atend to the history directly. 

in inference mode, it has memory bank, each input will attend to history and the k and v will be saved to the memroy bank, which is for the future text to attend.

below is the detailed plan
  
  
  # Refactor Plan: Unified Memory Path for Training and Inference

  ## Summary

  Replace the current stateful training prefill design with a single memory formulation used in both training and inference:

  - history and present use the same shared Qwen backbone weights
  - memory entries are always produced by TurnSummaryCompressor + write projections
  - present tokens always query memory entries through attention-style Q -> K,V retrieval
  - training builds temporary memory slots from batch history turns
  - inference keeps the same slot format in a persistent bank across turns

  This removes the current mismatch where training writes memory under no_grad while inference uses a different lifecycle.

  ## Implementation Changes

  ### 1. Redefine memory as layer-local compressed slots

  Adopt one memory entry = one completed turn summary, produced per selected IMM layer.

  - Choose per-layer bank as the architecture default.
  - Each selected IMM layer owns:
      - TurnSummaryCompressor
      - write projections to memory K and V
      - query projection from present hidden states
      - output projection / merge path
  - Remove training dependence on mutable MultiScopeMemoryState as the primary mechanism.
  - Keep memory semantics as session memory only for v1.
  - Drop working memory from the core training path; do not preserve it unless there is a separate proven use case.

  ### 2. Replace “prefill memory under no_grad” with dual-stream shared-backbone forward

  Refactor the model path so training uses two streams through the same backbone, not two separately owned model copies.

  - History stream:
      - encode each prior turn with the same backbone layers
      - at each selected IMM layer, compress each history turn hidden state with TurnSummaryCompressor
      - project summaries into layer-local K,V memory slots
  - Present stream:
      - encode prompt + current input + target tokens with the same backbone layers
      - at each selected IMM layer, compute Q from present hidden states
      - attend only to the compressed history memory slots, not to raw history tokens
      - merge retrieved memory into present hidden states, then continue to next layer
  - Important implementation rule:
      - this is one shared backbone executed on two streams
      - do not instantiate duplicated base models per layer
      - parameter sharing must come from reusing the same modules in two forward paths

  ### 3. Make train and infer memory format identical

  Define one canonical slot representation and use it everywhere.

  Canonical slot format:

  - scope: session only
  - granularity: one slot per completed turn, per selected IMM layer
  - contents:
      - compressed summary from TurnSummaryCompressor
      - projected K
      - projected V
      - optional metadata: turn_index, validity mask

  Training behavior:

  - for each sample, build layer-local memory slots from prior turns in the batch example
  - memory exists only for the duration of that forward pass
  - no persistent bank object is mutated during training
  - present tokens query those slots with attention-style retrieval

  Inference behavior:

  - maintain persistent per-session, per-layer banks of the same K,V slots
  - when generating a new response, present tokens query the current bank
  - after the assistant response completes, run the completed turn through the same backbone and the same
    TurnSummaryCompressor / write projections, then append one new slot per selected layer
  - use the same slot capacity and replacement policy as before if bounded memory is still needed

  ### 4. Reshape the code around explicit training/inference memory builders

  Refactor responsibilities rather than patching the current prefill helpers.

  Core code changes:

  - In modeling_imm.py:
      - replace the current wrapper behavior that reads/writes mutable runtime state inside layer forward
      - add explicit layer logic for:
          - building memory slots from history hidden states
          - querying memory slots from present hidden states
          - merging retrieved values into present hidden states
  - In train.py:
      - remove prefill_history_memory(...)
      - replace the single-pass train step with a dual-stream forward entrypoint
      - keep loss only on the present target tokens
  - In infer_tools.py:
      - change session state from generic runtime memory tensors to canonical per-layer slot banks
      - write one new turn slot after generation using the same write path used by training

  Interface changes:

  - introduce a dedicated memory-slot container type, separate from the old mutable read/write state API
  - model forward should accept explicit history_* tensors and present_* tensors for training
  - inference should use a dedicated method for “query with existing bank” and a dedicated method for “append completed turn
    to bank”

  ### 5. Adjust data representation to match the new forward

  Keep the dataset aligned with the new architecture.

  - Preserve history as turn-separated inputs, not flattened raw history text.
  - Training sample should provide:
      - history turns
      - present prompt + current input + target
      - supervision mask / labels for present tokens only
  - Do not let the present branch directly attend to history tokens through the base attention mask.
  - Keep history_lookup_mask only if needed for prompt-vs-target gating inside the present branch; otherwise simplify it away
    if memory querying should be allowed uniformly over the present segment.



  ## Assumptions And Defaults

  - Default architecture: per-layer memory bank.
  - Default write unit: one completed turn summary per selected IMM layer.
  - Default memory mechanism: attention-style retrieval where present hidden states query stored slots.
  - Default scope for v1: session memory only; working memory is out of scope.
  - Default write timing in inference: append memory after the full assistant response is generated, so training and
    inference both write completed turns.
  - Default pooling: reuse the existing TurnSummaryCompressor for both training and inference without introducing a second
    summary mechanism.


