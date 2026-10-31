import pytest

from app import profiles as p


def test_weights_gib_counts_packed_int4_as_i32():
    # the Hub reports 8 INT4 params per packed I32 word, so I32 weighs 0.5 byte per counted param
    gib = p.weights_gib({"I32": 31_406_948_352, "BF16": 4_523_902_832, "F32": 20_971_520})
    assert gib == pytest.approx(23.1, abs=1.0)


def test_weights_gib_unknown():
    assert p.weights_gib(None) is None
    assert p.weights_gib({}) is None


@pytest.mark.parametrize("w,verdict", [(5, "fits"), (17.8, "tight"), (23, "too_big")])
def test_fit_verdict_on_24gb(w, verdict):
    assert p.fit_verdict(w, 24.0)["verdict"] == verdict


def test_fit_verdict_edge_cases():
    assert p.fit_verdict(None, 24)["verdict"] == "unknown"
    assert p.fit_verdict(5, None)["verdict"] == "unknown"
    assert p.fit_verdict(5, 24, ["gguf"])["verdict"] == "unsupported"


def test_choose_util_leaves_room_for_the_os():
    assert p.choose_util(24564, 1500) == 0.90
    assert p.choose_util(24564, 100) == 0.90  # capped
    assert p.choose_util(24564, 6000) == 0.90  # idle WDDM usage is not real pressure
    assert p.choose_util(24564, 14000) == 0.36
    assert p.choose_util(24564, 24000) == 0.85  # implausibly low free memory falls back


def test_choose_context_scales_with_headroom():
    small = p.choose_context(5, 24, 0.88)
    big = p.choose_context(17.8, 24, 0.88)
    assert small > big >= 4096
    assert p.choose_context(None, 24, 0.88) == 8192
    assert p.choose_context(30, 24, 0.88) == 4096


@pytest.mark.parametrize("model,flag,value", [
    ("cyankiwi/Qwen3.6-27B-AWQ-INT4", "--tool-call-parser", "qwen3_coder"),
    ("Qwen/Qwen3-8B", "--reasoning-parser", "qwen3"),
    ("Qwen/Qwen2.5-Coder-7B-Instruct", "--tool-call-parser", "hermes"),
    ("mistralai/Devstral-Small-2505", "--tool-call-parser", "mistral"),
    ("meta-llama/Llama-3.1-8B-Instruct", "--tool-call-parser", "llama3_json"),
    ("openai/gpt-oss-20b", "--tool-call-parser", "openai"),
    ("some/unknown-model", "--tool-call-parser", "hermes"),
])
def test_model_flags(model, flag, value):
    flags = p.model_flags(model)
    assert flags[flags.index(flag) + 1] == value


def test_qwen3_coder_is_not_treated_as_plain_qwen3():
    assert "--reasoning-parser" not in p.model_flags("Qwen/Qwen3-Coder-30B-A3B-Instruct")


def test_build_vllm_args():
    args = p.build_vllm_args("Qwen/Qwen3-8B", 16384, 0.876)
    assert args[:4] == ["--model", "Qwen/Qwen3-8B", "--served-model-name", "qwen3-8b"]
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.88"
    assert "--enforce-eager" in args
    assert "--enforce-eager" not in p.build_vllm_args("Qwen/Qwen3-8B", 8192, 0.8, graphs=True)


def test_select_files_skips_duplicate_weights():
    files = ["config.json", "model-00001-of-00002.safetensors", "pytorch_model.bin",
             "consolidated.safetensors", "original/consolidated.00.pth", "model.gguf", "tokenizer.json"]
    assert p.select_files(files) == ["config.json", "model-00001-of-00002.safetensors", "tokenizer.json"]


def test_select_files_keeps_consolidated_when_only_weights():
    assert "consolidated.safetensors" in p.select_files(["consolidated.safetensors", "params.json"])


def test_quant_label():
    assert p.quant_label([], "cyankiwi/Qwen3.6-27B-AWQ-INT4") == "AWQ"
    assert p.quant_label([], "Qwen/Qwen3-8B") == "BF16/FP16"
