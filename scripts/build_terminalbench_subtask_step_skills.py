import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SKILLS = ROOT / "terminal_bench_skills_gen_0704.json"
OUTPUT_SKILLS = ROOT / "terminal_bench_skills_gen_0704_subtask_step.json"
DEEPSEEK_DIR = ROOT / "data/eval/deepseek_v4_pro_planner_terminalbench_20260702_2308_remote"
QWEN_DIR = ROOT / "data/eval/deepseek_v4_pro_qwen3_8b_router_terminalbench_20260703_1838_remote"


SUBTASK_INJECTION_POLICY = {
    "recommended_location": "initial_subtask_user_prompt",
    "lifetime": "entire_subtask",
    "top_k": 2,
    "notes": "Inject once when the subagent starts a delegated subtask. Keep visible through message history.",
}

STEP_INJECTION_POLICY = {
    "recommended_location": "temporary_system_prompt_suffix_in_live_messages",
    "lifetime": "next_llm_call_only",
    "top_k": 1,
    "notes": "Append to the live system prompt for the current step only; do not persist it in long-term message history.",
}


STEP_SKILL_SPECS = {
    "bn-fit-modify": [
        (
            "Check Python dependency imports before running full BN pipeline",
            "Before launching the full Bayesian-network modification workflow, run small import and API checks for pgmpy, pandas, numpy, and local scripts so dependency or version issues are fixed early.",
            ["Python probabilistic-graphical-model tasks", "Tasks with fragile third-party APIs", "Long pipelines where a full run is expensive"],
            ["Run minimal import checks first.", "Inspect expected constructor/function signatures before editing.", "Only run the full script after the smoke check passes."],
        ),
        (
            "Verify intervention output files with schema and statistics checks",
            "After producing intervention or modified-network artifacts, validate file existence, column/schema shape, and simple distribution statistics before submitting.",
            ["Generated CSV/JSON artifacts", "Causal intervention outputs", "Tasks graded by hidden validators"],
            ["Check all required output paths.", "Validate schema and row counts.", "Compute small sanity statistics and compare with task expectations."],
        ),
    ],
    "build-pmars": [
        (
            "Remove X11 compile and link flags before building headless source",
            "When a legacy C project must build in a headless Terminal-Bench container, identify display-library dependencies and replace them with console-safe compile options.",
            ["Legacy C builds", "Headless container build failures", "Makefile link errors involving X11 or graphics libraries"],
            ["Search Makefiles for GUI flags.", "Remove or stub display-only targets.", "Rebuild from clean state and capture the first compiler error."],
        ),
        (
            "Validate command-line game binary with verifier-like relative paths",
            "Run the built binary from the expected working directory with relative paths similar to the verifier, not only from the source directory.",
            ["Compiled CLI tools", "Build tasks with generated binaries", "Path-sensitive verifier failures"],
            ["Locate the final binary.", "Run it from the task root.", "Check exit code and expected stdout/stderr behavior."],
        ),
    ],
    "crack-7z-hash": [
        (
            "Extract archive hashes with the matching John helper before cracking",
            "Use the archive-specific hash extractor before invoking John or hashcat so the cracking tool receives the exact supported hash format.",
            ["Password-recovery tasks", "7z/zip/rar protected archives", "Hash format errors"],
            ["Identify archive type.", "Run the matching *2john helper.", "Confirm the resulting hash is accepted before spending time cracking."],
        ),
        (
            "Write and byte-check recovered secret output exactly",
            "After recovering a password or secret, write the required answer file with exact bytes and no extra newline unless the task asks for one.",
            ["Secret recovery tasks", "Exact-output graders", "Tasks requiring a final token file"],
            ["Inspect task output filename.", "Write only the recovered value.", "Re-read the file in binary mode to check bytes."],
        ),
    ],
    "distribution-search": [
        (
            "Switch distribution parameterization when KL constraints stagnate",
            "If optimization repeatedly misses KL or probability constraints, change the distribution parameterization instead of only tuning iteration counts.",
            ["Numerical search tasks", "Probability distribution fitting", "KL-divergence constraints"],
            ["Compute current constraint violations.", "Try an alternative parameter family.", "Keep the candidate with independently verified objective values."],
        ),
        (
            "Verify large probability files with independent KL recomputation",
            "Before submit, reload generated probabilities and recompute normalization and KL constraints with a separate short script.",
            ["Generated probability vectors", "Floating-point validators", "Large numeric output files"],
            ["Load output from disk.", "Check non-negativity and sum.", "Recompute all task metrics independently."],
        ),
    ],
    "dna-insert": [
        (
            "Parse complete FASTA records and locate exact insertion coordinates",
            "Use a real FASTA parser or record-aware code so sequence headers, wrapping, and insertion coordinates are handled exactly.",
            ["Bioinformatics sequence editing", "FASTA inputs with multiple records", "Coordinate-sensitive insertion tasks"],
            ["Parse records, not raw lines.", "Locate the target record and coordinate.", "Preserve headers and wrapping only after sequence edits are correct."],
        ),
        (
            "Run primer-specific validation before writing final FASTA",
            "After modifying DNA, verify target subsequences, primers, and expected lengths before writing the final FASTA.",
            ["Primer insertion tasks", "Sequence mutation tasks", "Hidden biological validators"],
            ["Search for required primer motifs.", "Check sequence length deltas.", "Write output only after validation passes."],
        ),
    ],
    "extract-elf": [
        (
            "Use ELF header offsets directly instead of unreliable npm parsers",
            "When existing parsers fail or misread an ELF, parse program and section headers directly using struct offsets from the ELF specification.",
            ["Binary forensics", "ELF extraction tasks", "Parser incompatibility failures"],
            ["Read the ELF identification bytes.", "Decode header tables with correct endianness.", "Use offsets from headers rather than guessed strings."],
        ),
        (
            "Bounds-check every binary read while mapping virtual addresses to file offsets",
            "Map virtual addresses through program headers and reject reads that fall outside the segment or file bounds.",
            ["ELF virtual-address lookups", "Segment extraction", "Binary patch or recovery tasks"],
            ["Find the containing segment.", "Compute file offset from vaddr.", "Check offset and length before reading."],
        ),
    ],
    "financial-document-processor": [
        (
            "Persist OCR text per document before extracting fields",
            "Save OCR output for each document as an intermediate artifact so extraction errors can be inspected and corrected without rerunning OCR blindly.",
            ["PDF/image financial documents", "OCR-dependent extraction", "Multi-document processors"],
            ["Run OCR per file.", "Store text beside the source.", "Inspect low-confidence or missing fields before parsing."],
        ),
        (
            "Validate extracted financial CSV against source documents",
            "Cross-check key fields such as dates, totals, vendor names, and currency formatting against OCR text before submitting a CSV.",
            ["Financial CSV generation", "Invoice/statement extraction", "Field-level hidden graders"],
            ["Load generated CSV.", "Compare each row to source text.", "Normalize only after preserving numeric meaning."],
        ),
    ],
    "fix-git": [
        (
            "Recover missing work by inspecting reflog before changing history",
            "Before reset, rebase, or checkout operations, inspect reflog and dangling commits to identify recoverable work.",
            ["Git recovery tasks", "Lost commit or branch problems", "Repository history repair"],
            ["Run status/log/reflog.", "Create a temporary recovery branch.", "Only then apply history edits."],
        ),
        (
            "Resolve merge conflicts by preserving verified user changes",
            "During conflict resolution, inspect both sides and tests so useful changes are preserved instead of selecting one side mechanically.",
            ["Merge conflict tasks", "Git repair with tests", "Multi-branch reconstruction"],
            ["Open conflicted files.", "Understand both versions.", "Run tests after resolving and before final commit/state."],
        ),
    ],
    "headless-terminal": [
        (
            "Inspect abstract terminal interface before implementing PTY behavior",
            "Read the expected terminal abstraction and tests before implementing PTY logic so method names, blocking behavior, and cleanup match the harness.",
            ["Headless terminal implementations", "PTY wrappers", "Interactive shell tasks"],
            ["Read interface/tests first.", "Implement the minimal required methods.", "Add cleanup for processes and file descriptors."],
        ),
        (
            "Verify PTY shell interactions across echo, interrupts, and interactive stdin",
            "Run interaction checks that cover command echo, stdin writes, Ctrl-C behavior, and process exit handling.",
            ["PTY behavior validation", "Interactive command agents", "Terminal emulator tests"],
            ["Start a shell.", "Send simple and interrupting commands.", "Assert output and exit behavior."],
        ),
    ],
    "kv-store-grpc": [
        (
            "Match gRPC service class names to generated stubs and task spec",
            "Inspect generated pb2_grpc files and the task spec before implementing server classes so method names and request fields align exactly.",
            ["Python gRPC services", "Generated protobuf stubs", "RPC method mismatch errors"],
            ["Regenerate or inspect stubs.", "Implement the exact servicer class.", "Use request/response field names from generated code."],
        ),
        (
            "Use portable Python socket and client checks for service readiness",
            "Validate gRPC readiness with a small Python client or socket check instead of relying on unavailable networking tools.",
            ["Minimal containers", "Service startup checks", "gRPC tasks without nc/curl"],
            ["Start server in background.", "Probe port with Python.", "Call at least one real RPC before submit."],
        ),
    ],
    "largest-eigenval": [
        (
            "Benchmark numerical replacements against the exact reference bottleneck",
            "Profile the reference implementation and verify any optimized numerical replacement against the original on small matrices.",
            ["Numerical optimization tasks", "Eigenvalue computations", "Performance-sensitive scientific code"],
            ["Profile first.", "Create small reference comparisons.", "Only optimize the measured bottleneck."],
        ),
        (
            "Use compiled helper only after verifying ABI and fallback behavior",
            "If adding compiled acceleration, verify importability, ABI compatibility, and a Python fallback path before depending on it.",
            ["C/C++/Cython numerical helpers", "Containers with uncertain compilers", "Scientific performance tasks"],
            ["Build in the target environment.", "Run import and sample calls.", "Keep or test a fallback when compilation fails."],
        ),
    ],
    "mcmc-sampling-stan": [
        (
            "Install and verify R packages with isolated library paths",
            "Use an explicit R library path and import checks so package installation state is reproducible inside the task container.",
            ["R/Stan tasks", "Package installation failures", "Containers with restricted libraries"],
            ["Create or inspect .libPaths().", "Install missing packages there.", "Run library() checks before executing the sampler."],
        ),
        (
            "Validate generated sampling outputs before finishing",
            "Check that Stan/R sampling outputs exist, have expected columns, and contain plausible finite values before submit.",
            ["MCMC output generation", "Stan model execution", "Statistical artifact validators"],
            ["Run the sampler.", "Load generated CSV/RDS output.", "Check dimensions, column names, and finite summaries."],
        ),
    ],
    "nginx-request-logging": [
        (
            "Rewrite nginx configuration cleanly after escaping failures",
            "When shell escaping corrupts nginx config edits, rewrite the affected block explicitly and validate syntax before restarting.",
            ["Nginx config tasks", "Log format edits", "Shell quoting failures"],
            ["Open the config file.", "Replace the whole relevant block.", "Run nginx -t before restart."],
        ),
        (
            "Verify web server behavior and custom logs end-to-end",
            "After config changes, send real HTTP requests and inspect the custom access log to verify format, fields, and rotation paths.",
            ["Web server logging tasks", "HTTP service validators", "Custom log format requirements"],
            ["Start/reload nginx.", "Send representative requests.", "Read the exact log file expected by the grader."],
        ),
    ],
    "portfolio-optimization": [
        (
            "Replace TODOs in native extension code only after reading tests",
            "Read the tests and native-extension build files before changing TODOs so function signatures and numeric tolerances match the harness.",
            ["Optimization code with TODOs", "Native extension tasks", "Test-driven numeric implementations"],
            ["Read tests first.", "Inspect build config and exported symbols.", "Implement only required TODO behavior."],
        ),
        (
            "Submit promptly after official tests pass and artifacts are present",
            "Once tests and required generated artifacts pass, submit instead of continuing risky refactors in a working solution.",
            ["Terminal-Bench tasks with submit command", "Long-running coding tasks", "Fragile verifier environments"],
            ["Run official tests.", "Check required files exist.", "Call submit immediately after passing state is reached."],
        ),
    ],
    "cobol-modernization": [
        (
            "Retrieve complete legacy source and data files before translating",
            "Before modernizing COBOL or legacy code, inventory every source, copybook, and data file so translation uses the full program context.",
            ["COBOL modernization", "Legacy batch programs", "Translation tasks with data files"],
            ["List source/data files.", "Identify entry points and copybooks.", "Translate only after the full dependency set is known."],
        ),
        (
            "Compare modernized output against legacy behavior on representative inputs",
            "Run the original or inferred legacy behavior on sample inputs and compare outputs from the modernized implementation.",
            ["Modernization validators", "Behavior-preserving rewrites", "Legacy-to-Python translations"],
            ["Create representative inputs.", "Run legacy/reference behavior where possible.", "Diff modern output and fix semantic gaps."],
        ),
    ],
    "count-dataset-tokens": [
        (
            "Enumerate dataset configs and splits before token counting",
            "Inspect available dataset configs and splits before counting so no required subset is skipped.",
            ["HuggingFace dataset tasks", "Token counting tasks", "Multi-config datasets"],
            ["List configs/splits.", "Select exactly those required by the prompt.", "Record counts per split before aggregation."],
        ),
        (
            "Combine all required text fields before applying tokenizer counts",
            "Build token-count input strings from every prompt-required text field in the correct order before calling the tokenizer.",
            ["Dataset token accounting", "Structured records with multiple text fields", "Tokenizer-sensitive graders"],
            ["Inspect one record schema.", "Concatenate fields per spec.", "Use the specified tokenizer and aggregation rule."],
        ),
    ],
    "modernize-scientific-stack": [
        (
            "Map old scientific APIs to modern equivalents with smoke tests",
            "Replace deprecated scientific-library APIs by mapping each call to a modern equivalent and testing it on small representative inputs.",
            ["SciPy/NumPy modernization", "Deprecated API failures", "Scientific Python stack upgrades"],
            ["Locate failing imports/calls.", "Map to current APIs.", "Run small smoke tests after each replacement."],
        ),
        (
            "Verify modernized scientific outputs against baseline invariants",
            "After modernization, compare shapes, units, monotonicity, and numeric tolerances against expected invariants.",
            ["Scientific code rewrites", "Numerical compatibility tasks", "Hidden tests checking behavior"],
            ["Run available tests.", "Check domain invariants.", "Avoid changing algorithms beyond compatibility fixes."],
        ),
    ],
    "password-recovery": [
        (
            "Reconstruct deleted binary fragments using offsets and neighboring bytes",
            "Use offsets, file signatures, and neighboring bytes to reconstruct missing binary fragments before attempting password extraction.",
            ["Forensic recovery tasks", "Deleted or corrupted files", "Binary fragment reconstruction"],
            ["Identify candidate fragments.", "Check magic bytes and offsets.", "Reassemble and validate the recovered file type."],
        ),
        (
            "Chain forensic tools and verify recovered secret exactly",
            "Use multiple recovery/cracking tools when needed, then verify the final secret by opening or decrypting the target artifact.",
            ["Password recovery", "Forensics pipelines", "Exact secret submission"],
            ["Extract candidate hashes/artifacts.", "Try appropriate recovery tools.", "Verify the secret against the original artifact."],
        ),
    ],
}


def find_trajectory(eval_dir: Path, task_id: str) -> str | None:
    for attempt in ("attempt_0", "attempt_1"):
        path = eval_dir / "logs" / attempt / task_id / "trajectory.json"
        if path.exists():
            return str(path.relative_to(ROOT))
    return None


def make_step_skill(task_id: str, spec_index: int, spec: tuple[str, str, list[str], list[str]]) -> dict:
    name, description, scenarios, principles = spec
    return {
        "name": name,
        "level": "step",
        "description": description,
        "application_scenarios": scenarios,
        "execution_principles": principles,
        "workflow": [
            "Inspect the immediate error, file, or artifact relevant to the current step.",
            "Run the smallest command that verifies the assumption behind this skill.",
            "Apply the targeted edit or command sequence.",
            "Re-run the local check and only then proceed to the next step.",
        ],
        "example": (
            f"In task `{task_id}`, use this only for the current execution step when the subagent is about "
            "to run commands matching the scenario; do not keep injecting it after the step is resolved."
        ),
        "failure_cases_prevented": [
            "Repeating broad exploration after the relevant operation is already known.",
            "Submitting artifacts that exist but fail exact verifier expectations.",
            "Using a generic command that ignores task-specific file formats or runtime constraints.",
        ],
        "skill_granularity": "step_level",
        "generality": "task_specific_or_domain_operation",
        "injection_policy": copy.deepcopy(STEP_INJECTION_POLICY),
        "source": {
            "task_id": task_id,
            "deepseek_success_eval_dir": str(DEEPSEEK_DIR.relative_to(ROOT)),
            "qwen_failure_eval_dir": str(QWEN_DIR.relative_to(ROOT)),
            "deepseek_trajectory": find_trajectory(DEEPSEEK_DIR, task_id),
            "qwen_trajectory": find_trajectory(QWEN_DIR, task_id),
            "distillation_basis": "derived by splitting successful DeepSeek trajectory behavior into current-step operational skills and contrasting with Qwen3-8B failure trajectory",
            "derived_from": "trajectory_and_command_events",
            "step_skill_index_within_task": spec_index + 1,
        },
    }


def main() -> None:
    original_skills = json.loads(SOURCE_SKILLS.read_text(encoding="utf-8"))
    if not isinstance(original_skills, list):
        raise TypeError(f"{SOURCE_SKILLS} must contain a JSON list")

    task_counts: dict[str, int] = {}
    task_skills = []
    for skill in original_skills:
        task_id = skill.get("source", {}).get("task_id")
        if not task_id:
            raise ValueError(f"skill lacks source.task_id: {skill.get('name')}")
        task_counts[task_id] = task_counts.get(task_id, 0) + 1

        normalized = copy.deepcopy(skill)
        normalized["level"] = "subtask"
        normalized["skill_granularity"] = "subtask_level"
        normalized["generality"] = "domain_or_task_strategy"
        normalized["injection_policy"] = copy.deepcopy(SUBTASK_INJECTION_POLICY)
        normalized.setdefault("source", {})
        normalized["source"].update(
            {
                "deepseek_success_eval_dir": str(DEEPSEEK_DIR.relative_to(ROOT)),
                "qwen_failure_eval_dir": str(QWEN_DIR.relative_to(ROOT)),
                "deepseek_trajectory": find_trajectory(DEEPSEEK_DIR, task_id),
                "qwen_trajectory": find_trajectory(QWEN_DIR, task_id),
                "distillation_basis": "original distilled skill kept as subtask-level strategy from DeepSeek success trajectory",
                "source_skill_index_within_task": task_counts[task_id],
            }
        )
        task_skills.append(normalized)

    expected_tasks = set(task_counts)
    missing_specs = sorted(expected_tasks - set(STEP_SKILL_SPECS))
    extra_specs = sorted(set(STEP_SKILL_SPECS) - expected_tasks)
    bad_counts = {task: len(specs) for task, specs in STEP_SKILL_SPECS.items() if len(specs) != 2}
    if missing_specs or extra_specs or bad_counts:
        raise ValueError(
            f"step spec mismatch: missing={missing_specs}, extra={extra_specs}, bad_counts={bad_counts}"
        )

    step_skills = []
    for task_id in sorted(expected_tasks):
        if task_counts[task_id] != 2:
            raise ValueError(f"task {task_id} has {task_counts[task_id]} subtask skills, expected 2")
        for index, spec in enumerate(STEP_SKILL_SPECS[task_id]):
            step_skills.append(make_step_skill(task_id, index, spec))

    output = {
        "schema_version": "terminal_bench_skills.subtask_step.v1",
        "created_from": str(SOURCE_SKILLS.relative_to(ROOT)),
        "do_not_overwrite_source_file": True,
        "source_eval_dirs": {
            "deepseek_success": str(DEEPSEEK_DIR.relative_to(ROOT)),
            "qwen3_8b_failure_baseline": str(QWEN_DIR.relative_to(ROOT)),
        },
        "design": {
            "subtask_level": "Strategy/process skills injected once at subtask start and kept through subagent message history.",
            "step_level": "Concrete operational skills injected temporarily into the current step system prompt and not persisted.",
            "per_task_counts": {"subtask_level": 2, "step_level": 2},
        },
        "task_skills": task_skills,
        "step_skills": step_skills,
    }
    OUTPUT_SKILLS.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_SKILLS}")
    print(f"tasks={len(expected_tasks)} subtask_skills={len(task_skills)} step_skills={len(step_skills)}")


if __name__ == "__main__":
    main()
