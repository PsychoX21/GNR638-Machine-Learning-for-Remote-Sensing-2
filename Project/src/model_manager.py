import os
import sys
import time
import signal
import base64
import subprocess
import logging
import re
import io
from typing import Optional, List, Dict, Any
from PIL import Image

logger = logging.getLogger(__name__)

PRIMARY_MODEL_PATH = "./weights/qwen35_27b_fp8"
VLLM_PORT = 8000
VLLM_HOST = "http://localhost"
VLLM_LOG_FILE = "vllm_server.log"


def _encode_image_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _strip_thinking_tags(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


class ModelManager:
    def __init__(self):
        self._vllm_process: Optional[subprocess.Popen] = None
        self._vllm_log_handle = None
        self._openai_client = None
        self._model_id: Optional[str] = None

    def start_primary_model(self) -> None:
        logger.info("Starting vLLM server for Qwen/Qwen3.5-27B-FP8...")

        model_path = PRIMARY_MODEL_PATH
        if not os.path.exists(model_path):
            model_path = "Qwen/Qwen3.5-27B-FP8"
            logger.warning(f"Local weights not found, using HF ID: {model_path}")

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--gpu-memory-utilization", "0.88",
            "--max-model-len", "16384",
            "--max-num-seqs", "1",
            "--dtype", "auto",
            "--port", str(VLLM_PORT),
            "--trust-remote-code",
            "--disable-log-stats",
            "--enforce-eager",
            "--reasoning-parser", "qwen3"
        ]

        logger.info(f"vLLM cmd: {' '.join(cmd)}")

        # Write vLLM output to a log file instead of PIPE to prevent
        # pipe buffer deadlock (vLLM produces heavy output)
        self._vllm_log_handle = open(VLLM_LOG_FILE, "w")

        self._vllm_process = subprocess.Popen(
            cmd,
            stdout=self._vllm_log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        self._wait_for_vllm_ready(timeout=900)

        from openai import OpenAI
        self._openai_client = OpenAI(
            base_url=f"{VLLM_HOST}:{VLLM_PORT}/v1",
            api_key="EMPTY",
        )

        try:
            self._model_id = self._openai_client.models.list().data[0].id
            logger.info(f"Model ID: {self._model_id}")
        except Exception as e:
            logger.warning(f"Could not fetch model ID: {e}")
            self._model_id = model_path

        logger.info("vLLM server ready.")

    def _wait_for_vllm_ready(self, timeout: int = 900) -> None:
        import urllib.request
        import urllib.error

        start = time.time()
        health_url = f"{VLLM_HOST}:{VLLM_PORT}/health"

        while time.time() - start < timeout:
            try:
                req = urllib.request.urlopen(health_url, timeout=5)
                if req.status == 200:
                    logger.info(f"vLLM ready after {time.time() - start:.1f}s")
                    return
            except (urllib.error.URLError, ConnectionRefusedError, OSError):
                pass

            if self._vllm_process.poll() is not None:
                log_tail = ""
                try:
                    with open(VLLM_LOG_FILE, "r") as f:
                        log_tail = f.read()[-3000:]
                except Exception:
                    pass
                raise RuntimeError(f"vLLM died. Log tail:\n{log_tail}")

            time.sleep(3)

        raise TimeoutError(f"vLLM not ready within {timeout}s")

    def call_primary(
        self,
        text_prompt: str,
        images: Optional[List[Image.Image]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        thinking_mode: bool = False,
    ) -> str:
        if self._openai_client is None:
            raise RuntimeError("Primary model not started.")

        think_tag = "/think" if thinking_mode else "/no_think"
        system_content = f"{think_tag}\nYou are a deep learning expert assistant."

        user_content = []
        if images:
            for img in images:
                img_b64 = _encode_image_base64(img)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
        user_content.append({"type": "text", "text": text_prompt})

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        try:
            response = self._openai_client.chat.completions.create(
                model=self._model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = response.choices[0].message.content or ""
            result = _strip_thinking_tags(result)
            return result
        except Exception as e:
            logger.error(f"Primary call failed: {e}")
            raise

    def call_secondary(
        self,
        text_prompt: str,
        images: Optional[List[Image.Image]] = None,
        max_tokens: int = 512,
    ) -> str:
        if self._openai_client is None:
            raise RuntimeError("Primary model not started.")

        system_content = (
            "/no_think\n"
            "You are a concise exam grader. Read the question and "
            "select the single best answer. Output ONLY the option number."
        )

        user_content = []
        if images:
            for img in images:
                img_b64 = _encode_image_base64(img)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
        user_content.append({"type": "text", "text": text_prompt})

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        try:
            response = self._openai_client.chat.completions.create(
                model=self._model_id,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            result = response.choices[0].message.content or ""
            return _strip_thinking_tags(result)
        except Exception as e:
            logger.error(f"Self-consistency call failed: {e}")
            return ""

    def stop_primary_model(self) -> None:
        if self._vllm_process is not None:
            logger.info("Shutting down vLLM...")
            try:
                os.killpg(os.getpgid(self._vllm_process.pid), signal.SIGTERM)
                self._vllm_process.wait(timeout=15)
            except Exception:
                try:
                    self._vllm_process.kill()
                except Exception:
                    pass
            self._vllm_process = None

        if self._vllm_log_handle is not None:
            try:
                self._vllm_log_handle.close()
            except Exception:
                pass
            self._vllm_log_handle = None

    def cleanup(self) -> None:
        self.stop_primary_model()
