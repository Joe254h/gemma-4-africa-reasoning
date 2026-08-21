# Reproducibility contract

A result is reproducible only when all of the following are available:

- Git commit;
- configuration file and checksum;
- model ID and immutable revision;
- dataset ID, immutable revision, selected indices, source-file checksums, and evaluation checksum;
- exact Python package versions;
- GPU name, VRAM, driver, CUDA build, and Slurm allocation;
- prompt template and decoding controls;
- raw prediction count and unique sample IDs;
- automatic scoring output;
- human-review denominator and reviewer qualification for low-resource languages.

The project pins application libraries in `requirements/hpc.txt`. PyTorch is installed separately because the correct wheel depends on the HPC driver and CUDA environment. Every run records the actual PyTorch version, and the Slurm job writes `environment.lock.txt` after execution. Once the GPU audit determines the cluster-compatible wheel, record its index URL and exact version in the experiment notes.

Generated evaluation data is committed because it is small, license-compatible, and necessary for exact before/after pairing. Raw model traces remain ignored until reviewed.

Never rerun a comparison after silently changing a prompt, maximum output length, model revision, dataset revision, or parser version. Create a new configuration and baseline instead.
