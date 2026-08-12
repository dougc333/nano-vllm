from huggingface_hub import snapshot_download

snapshot_download("Qwen/Qwen3-30B-A3B", local_dir="test_progs/Qwen3-30B-A3B")
snapshot_download("Qwen/Qwen3-30B-A3B", local_dir="test_progs/Qwen3-0.6")

