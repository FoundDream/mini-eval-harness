from __future__ import annotations

import json
import os
import importlib
from dataclasses import dataclass
from typing import Any, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ModelOutput:
    text: str
    raw: dict[str, object]


class ModelAdapter(Protocol):
    name: str

    def generate(self, prompt: str) -> ModelOutput:
        ...


class MockAdapter:
    name = "mock-v1"

    def generate(self, prompt: str) -> ModelOutput:
        output = self._answer(prompt)
        return ModelOutput(
            text=output,
            raw={
                "provider": "mock",
                "model": self.name,
                "output": output,
            },
        )

    @staticmethod
    def _answer(prompt: str) -> str:
        agent_answer = MockAdapter._answer_agent_demo(prompt)
        if agent_answer is not None:
            return agent_answer

        rag_answer = MockAdapter._answer_rag_demo(prompt)
        if rag_answer is not None:
            return rag_answer

        if "中国首都" in prompt or "中国的首都" in prompt:
            return "北京"
        if "1+1" in prompt:
            return "2"
        if "苹果" in prompt and "颜色" in prompt:
            return "红色"
        return "不知道"

    @staticmethod
    def _answer_agent_demo(prompt: str) -> str | None:
        if "AGENT_EVAL_DEMO" not in prompt:
            return None

        task_id = MockAdapter._extract_agent_field(prompt, "Task id:")
        step_count_text = MockAdapter._extract_agent_field(prompt, "Completed steps:")
        try:
            step_count = int(step_count_text)
        except ValueError:
            step_count = 0

        scripts: dict[str, list[dict[str, object]]] = {
            "agent_001": [
                {
                    "thought": "先查看 Alice 的年假余额。",
                    "tool": "get_user_profile",
                    "args": {"user_id": "alice"},
                },
                {
                    "thought": "余额少于 3 天，需要提醒。",
                    "tool": "send_message",
                    "args": {"user_id": "alice", "text": "你的年假余额还有 2 天，请及时规划。"},
                },
                {"thought": "提醒已发送。", "final_answer": "已提醒 Alice。"},
            ],
            "agent_002": [
                {
                    "thought": "先计算乘法结果。",
                    "tool": "calculator",
                    "args": {"expression": "37 * 18"},
                },
                {
                    "thought": "把结果写入待办备注并完成任务。",
                    "tool": "update_todo",
                    "args": {"item_id": "todo_calc", "status": "done", "note": "37 * 18 = 666"},
                },
                {"thought": "任务已完成。", "final_answer": "已完成 todo_calc。"},
            ],
            "agent_003": [
                {
                    "thought": "先搜索报销审批规则。",
                    "tool": "search_docs",
                    "args": {"query": "超过 1000 元 报销 审批", "top_k": 2},
                },
                {
                    "thought": "把结论发给 Bob。",
                    "tool": "send_message",
                    "args": {
                        "user_id": "bob",
                        "text": "单笔超过 1000 元的报销需要直属经理和财务两级审批。",
                    },
                },
                {"thought": "已通知 Bob。", "final_answer": "已通知 Bob。"},
            ],
            "agent_004": [
                {
                    "thought": "先搜索安全要求。",
                    "tool": "search_docs",
                    "args": {"query": "代码仓库 生产系统 多因素认证", "top_k": 2},
                },
                {
                    "thought": "把关键要求写入任务备注。",
                    "tool": "update_todo",
                    "args": {
                        "item_id": "security_review",
                        "status": "done",
                        "note": "访问代码仓库和生产系统必须启用多因素认证。",
                    },
                },
                {"thought": "安全要求已记录。", "final_answer": "已完成 security_review。"},
            ],
            "agent_005": [
                {
                    "thought": "先查看 Chen 的设备状态。",
                    "tool": "get_user_profile",
                    "args": {"user_id": "chen"},
                },
                {
                    "thought": "设备丢失，需要通知 IT。",
                    "tool": "send_message",
                    "args": {"user_id": "it", "text": "Chen 的设备已丢失，请协助处理。"},
                },
                {
                    "thought": "涉及设备丢失，也要通知安全团队。",
                    "tool": "send_message",
                    "args": {"user_id": "security", "text": "Chen 的设备已丢失，请同步安全响应。"},
                },
                {"thought": "通知已完成。", "final_answer": "已通知 IT 和 security。"},
            ],
            "agent_006": [
                {
                    "thought": "先计算总额。",
                    "tool": "calculator",
                    "args": {"expression": "1200 + 350"},
                },
                {
                    "thought": "把总额发送给财务。",
                    "tool": "send_message",
                    "args": {"user_id": "finance", "text": "报销总额为 1550。"},
                },
                {"thought": "已通知财务。", "final_answer": "已发送总额。"},
            ],
            "agent_007": [
                {
                    "thought": "先搜索新员工培训期限。",
                    "tool": "search_docs",
                    "args": {"query": "新员工 信息安全培训 完成期限", "top_k": 2},
                },
                {
                    "thought": "把期限提醒 Diana。",
                    "tool": "send_message",
                    "args": {
                        "user_id": "diana",
                        "text": "新员工需要在 3 个工作日内完成信息安全培训。",
                    },
                },
                {"thought": "提醒已发送。", "final_answer": "已提醒 Diana。"},
            ],
            "agent_008": [
                {
                    "thought": "先搜索住宿报销材料要求。",
                    "tool": "search_docs",
                    "args": {"query": "出差住宿 报销 材料 酒店发票 入住水单", "top_k": 2},
                },
                {
                    "thought": "缺少材料时应阻塞该任务。",
                    "tool": "update_todo",
                    "args": {
                        "item_id": "hotel_claim",
                        "status": "blocked",
                        "note": "出差住宿报销需要酒店发票和入住水单，缺少任一材料会被退回。",
                    },
                },
                {"thought": "任务状态已更新。", "final_answer": "已阻塞 hotel_claim。"},
            ],
            "agent_009": [
                {
                    "thought": "先查看 Emma 的密码年龄。",
                    "tool": "get_user_profile",
                    "args": {"user_id": "emma"},
                },
                {
                    "thought": "超过 90 天，需要提醒改密。",
                    "tool": "send_message",
                    "args": {"user_id": "emma", "text": "你的密码已超过 90 天，请尽快更新密码。"},
                },
                {"thought": "提醒已发送。", "final_answer": "已提醒 Emma。"},
            ],
            "agent_010": [
                {
                    "thought": "先查看 Frank 的待办负责人信息。",
                    "tool": "get_user_profile",
                    "args": {"user_id": "frank"},
                },
                {
                    "thought": "归档已完成任务。",
                    "tool": "update_todo",
                    "args": {"item_id": "archive_done", "status": "archived", "note": "已归档"},
                },
                {"thought": "任务已归档。", "final_answer": "已归档 archive_done。"},
            ],
        }

        script = scripts.get(task_id)
        if script is None:
            return json.dumps(
                {"thought": "未知任务。", "final_answer": "无法处理。"},
                ensure_ascii=False,
            )
        action = script[min(step_count, len(script) - 1)]
        return json.dumps(action, ensure_ascii=False)

    @staticmethod
    def _extract_agent_field(prompt: str, label: str) -> str:
        for line in prompt.splitlines():
            if line.startswith(label):
                return line.removeprefix(label).strip()
        return ""

    @staticmethod
    def _answer_rag_demo(prompt: str) -> str | None:
        if "企业制度问答助手" not in prompt:
            return None

        question = MockAdapter._extract_rag_question(prompt)
        if "试用期员工" in question and "年假" in question:
            return "试用期员工暂不享受年假。"
        if "正式员工" in question and "带薪年假" in question:
            return "正式员工每个自然年享有 10 天带薪年假。"
        if "病假超过" in question:
            return "连续病假超过 1 天时，员工需要上传医院证明或医生诊断材料。"
        if "年假最多可以结转" in question:
            return "未休完的年假最多可以结转 5 天到下一自然年。"
        if "报销需要在费用发生后" in question:
            return "员工报销需要在费用发生后的 30 天内提交申请。"
        if "超过 1000 元" in question or "超过1000元" in question:
            return "单笔金额超过 1000 元的报销，需要直属经理和财务两级审批。"
        if "工作打车" in question or "打车可以报销" in question:
            return "晚上 22:00 之后因工作原因产生的打车费用可以报销。"
        if "餐饮报销" in question and "酒水" in question:
            return "餐饮报销不得包含酒水，酒水部分需要自行承担。"
        if "信息安全培训" in question:
            return "新员工入职后需要在 3 个工作日内完成信息安全培训。"
        if "指定导师" in question:
            return "新员工的直属经理需要在入职第一周内指定一名导师。"
        if "试用期一般" in question:
            return "试用期一般为 3 个月。"
        if "入职材料缺失" in question or "多久内补交" in question:
            return "入职材料缺失时，员工需要在 5 个工作日内完成补交。"
        if "申请笔记本电脑" in question:
            return "员工申请笔记本电脑需要通过 IT 服务台提交设备申请单。"
        if "办公设备丢失" in question:
            return "办公设备丢失后，员工需要在 24 小时内向 IT 和直属经理报备。"
        if "个人设备" in question and "生产环境" in question:
            return "个人设备不得直接访问生产环境。"
        if "离职前" in question and "归还" in question:
            return "员工离职前需要归还笔记本电脑、显示器、门禁卡和其他公司资产。"
        if "代码仓库和生产系统" in question:
            return "未启用多因素认证的账号不能访问代码仓库和生产系统。"
        if "更新一次登录密码" in question or "登录密码" in question:
            return "员工需要每 90 天更新一次登录密码。"
        if "客户数据" in question and "对外共享" in question:
            return "包含客户数据的文件对外共享前，必须获得信息安全团队审批。"
        if "钓鱼邮件" in question or "可疑链接" in question:
            return "发现钓鱼邮件或可疑链接时，员工需要立即转发给安全响应邮箱，并避免点击链接或下载附件。"
        return "资料不足，无法回答。"

    @staticmethod
    def _extract_rag_question(prompt: str) -> str:
        if "问题：" not in prompt:
            return prompt
        question_region = prompt.split("问题：", 1)[1]
        if "资料：" in question_region:
            question_region = question_region.split("资料：", 1)[0]
        return question_region.strip()


class OpenAIChatAdapter:
    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_seconds: float = 60.0,
        extra_body: dict[str, object] | None = None,
    ) -> None:
        self.name = model
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.extra_body = extra_body or {}

    def generate(self, prompt: str) -> ModelOutput:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {self.api_key_env}")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        payload.update(self.extra_body)
        response = self._post_json("/chat/completions", payload, api_key)
        choices = cast(list[dict[str, Any]], response.get("choices", []))
        if not choices:
            raise RuntimeError("OpenAI-compatible response has no choices")

        message = cast(dict[str, Any], choices[0].get("message", {}))
        text = message.get("content")
        if text is None:
            raise RuntimeError("OpenAI-compatible response has no message.content")

        return ModelOutput(text=str(text), raw=response)

    def _post_json(
        self, path: str, payload: dict[str, object], api_key: str
    ) -> dict[str, object]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {error_body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Request failed for {url}: {exc.reason}") from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON response from {url}") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError(f"Unexpected JSON response type from {url}")
        return parsed


class HFTransformersAdapter:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        use_chat_template: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            torch: Any = importlib.import_module("torch")
            transformers: Any = importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "HFTransformersAdapter requires transformers and torch. "
                'Install them with: pip install -e ".[hf]"'
            ) from exc
        AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")
        AutoTokenizer = getattr(transformers, "AutoTokenizer")

        self.name = model_name_or_path
        self.model_name_or_path = model_name_or_path
        self.device = self._resolve_device(torch, device)
        self.dtype_name = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_chat_template = use_chat_template
        self.trust_remote_code = trust_remote_code
        self._torch = torch

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, object] = {
            "trust_remote_code": trust_remote_code,
        }
        torch_dtype = self._resolve_dtype(torch, dtype)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        model.to(self.device)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model

    def generate(self, prompt: str) -> ModelOutput:
        torch = self._torch
        inputs = self._build_inputs(prompt)
        prompt_token_count = int(inputs["input_ids"].shape[-1])
        generation_kwargs: dict[str, object] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.tokenizer.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        generated_ids = output_ids[0][prompt_token_count:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return ModelOutput(
            text=text,
            raw={
                "provider": "hf",
                "model": self.model_name_or_path,
                "device": self.device,
                "dtype": self.dtype_name,
                "prompt_tokens": prompt_token_count,
                "completion_tokens": int(generated_ids.shape[-1]),
                "use_chat_template": self.use_chat_template,
            },
        )

    def _build_inputs(self, prompt: str) -> dict[str, Any]:
        if self.use_chat_template:
            if not hasattr(self.tokenizer, "apply_chat_template"):
                raise RuntimeError("Tokenizer does not support apply_chat_template")
            messages = [{"role": "user", "content": prompt}]
            try:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            except TypeError:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )

            if hasattr(encoded, "items") and "input_ids" in encoded:
                inputs = {
                    key: value.to(self.device)
                    for key, value in encoded.items()
                    if hasattr(value, "to")
                }
                if "input_ids" not in inputs:
                    raise RuntimeError("Chat template did not return input_ids")
                if "attention_mask" not in inputs:
                    inputs["attention_mask"] = self._torch.ones_like(
                        inputs["input_ids"]
                    ).to(self.device)
                return inputs

            input_ids = encoded
            input_ids = input_ids.to(self.device)
            return {
                "input_ids": input_ids,
                "attention_mask": self._torch.ones_like(input_ids).to(self.device),
            }

        inputs = self.tokenizer(prompt, return_tensors="pt")
        return {key: value.to(self.device) for key, value in inputs.items()}

    @staticmethod
    def _resolve_device(torch: Any, device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_dtype(torch: Any, dtype: str) -> object | None:
        if dtype == "auto":
            return None
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(
                f"Unsupported HF dtype: {dtype}. "
                "Use auto, float16, bfloat16, or float32."
            )
        return dtype_map[dtype]
