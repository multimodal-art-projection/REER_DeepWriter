# REverse-Engineered Reasoning for Open-Ended Generation
The official code of "REverse-Engineered Reasoning for Open-Ended Generation"

## Release Progress
- [x] Deep Reasoning Synthesis

- [ ] evaluation

- [ ] VeRL based Distributed SFT training

## Synthesis of Deep Reasoning
cd folder: `synthesize_deep_reasoning`

- **Step 0: Update the config.**

  check `config.yaml`:
    ```
      stop_thresh: 0.25 # PPL stopping criterion
      max_step: 10 # max-step stopping criterion
      num_rollouts: 1 # num initial thinking rollouts each query, not tested
      num_expansion: 2 # num expanded node for each segment edits
      file_path: '/path/to/QAcollection.json'
      file_prefix: '/path/to/output/file/folder/'
    ```
  Json format: a list of dicts, where each dict has three keys, `question`, `solution`, `index`

- **Step 1: Start the vLLM server.**
  ```bash 
  export model=/path/to/generator
  export model2=/path/to/basemodel/for/PPL
  bash server.sh
  ```
  We use Qwen2.5-32B-Instruct as the generator, and Qwen3-8B-Base as the model for computing perplexity. We find it faster if we amortize the PPL computation to a smaller model. 

- **Step 2: Run the Deep Reasoning Synthesis with Ray-scheduled multi-workers.**
  ```
  export workdir=${pwd}
  export model=/path/to/generator
  export model2=/path/to/basemodel/for/PPL
  export port=2233
  export rank=0 
  export total=1
  export cname=/path/to/config
  bash synthesis.sh
  ```
  The synthesized trajectories will be dumped to the `file_prefix` path. 
