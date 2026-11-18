"""Silence third-party library chatter so the demos print only our output.

Import this FIRST (before torch/transformers) in the demo entry points. Pass
--verbose to a demo to skip it. Nothing here changes behaviour, only logging:
progress bars, "experimental backend" notices, deprecation warnings inside
libraries, and the Hugging Face "unauthenticated requests" hint.
"""
import logging
import os
import sys
import warnings

if "--verbose" not in sys.argv:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    os.environ.setdefault("CT2_VERBOSE", "-1")          # CTranslate2: errors only
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    for name in ("torch.distributed.elastic.multiprocessing.redirects", "huggingface_hub", "transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from huggingface_hub.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass
else:
    sys.argv.remove("--verbose")
