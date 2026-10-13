# -*- coding: utf-8 -*-
"""三层环境检测（L1/L2/L3）的契约测试。

L1 ``check_libs()`` —— 只读 ``importlib.metadata`` 的静态库检测
L2 ``probe_device()`` —— 子进程里的动态设备探测
L3 ``build_report_rows()`` / ``diagnostic_text()`` —— 给 GUI 面板的数据

★这些测试断言的是**契约**（字段、级别取值、remedy 可操作性），
不断言具体库版本或硬件型号——换台机器就得全改一遍的那种测试没有价值。
"""

import subprocess
import sys
import time

import pytest

from audio_separation.core.env_check import (
    build_report_rows,
    check_libs,
    diagnostic_text,
    probe_device,
)
from audio_separation.core.requirements import REQUIREMENTS

#: check_libs() 里真实使用的全部级别取值
LEVELS = {"PASS", "FAIL", "WARN", "INFO"}
#: 需要用户动手处理的级别
ACTIONABLE = {"FAIL", "WARN"}


class TestL1CheckLibs:
    def test_returns_one_row_per_requirement(self, env_report):
        assert len(env_report) == len(REQUIREMENTS)

    def test_row_schema(self, env_report):
        for row in env_report:
            missing = {"item", "level", "detail", "remedy"} - set(row)
            assert not missing, f"{row.get('item', '?')} 缺字段：{missing}"
            assert row["item"], "item 不得为空"
            assert row["level"] in LEVELS, f"{row['item']} 级别非法：{row['level']}"
            assert isinstance(row["detail"], str), f"{row['item']} 的 detail 必须是字符串"

    def test_remedy_is_str_or_backend_dict(self, env_report):
        """torch 缺失时 remedy 是 {后端: 命令} 字典，其余条目是单个字符串。"""
        for row in env_report:
            remedy = row["remedy"]
            if isinstance(remedy, dict):
                assert remedy and all(
                    isinstance(k, str) and isinstance(v, str) and v.strip()
                    for k, v in remedy.items()
                ), f"{row['item']} 的 remedy 字典含非法条目：{remedy}"
            else:
                assert isinstance(remedy, str), f"{row['item']} 的 remedy 类型异常：{type(remedy)}"

    def test_pass_rows_state_what_version_was_found(self, env_report):
        """PASS 行必须说清查到了什么版本，否则面板上一片绿却看不出深浅。"""
        for row in env_report:
            if row["level"] == "PASS":
                assert row["detail"].strip(), f"{row['item']} 报 PASS 却没有版本信息"

    def test_actionable_rows_carry_a_remedy(self, env_report):
        """★FAIL/WARN 行必须附可复制的修复命令——这是 §2.6 的核心承诺。

        给了「红色告警」却不告诉人怎么办，等于把问题原样甩回给用户。
        """
        missing = [
            row["item"]
            for row in env_report
            if row["level"] in ACTIONABLE and not row["remedy"]
        ]
        assert not missing, f"这些条目报了 {ACTIONABLE} 却没有给出修复命令：{missing}"

    def test_torch_row_is_labelled_detect_only(self, env_report):
        """torch 行必须自带「只检测不安装」字样，让用户知道工具不会替他装。"""
        torch_row = next(r for r in env_report if r["item"].startswith("torch"))
        assert "只检测不安装" in torch_row["item"]

    def test_is_fast(self):
        """L1 必须秒级。

        超过 10 s 就说明有人把「查元数据」改成了「真 import」，
        而本机 import bs_roformer 会挂死 >120 s。
        """
        start = time.perf_counter()
        check_libs()
        elapsed = time.perf_counter() - start
        assert elapsed < 10, f"check_libs() 耗时 {elapsed:.2f}s，超过 10s"

    def test_leaves_heavy_modules_unimported(self):
        """★在干净子进程里跑一次，确认 L1 没有把 torch/bs_roformer 拉进来。

        同进程里断言 ``"torch" not in sys.modules`` 不可靠——
        pytest 插件或别的测试可能早就 import 过。
        """
        code = (
            "import sys;"
            "from audio_separation.core.env_check import check_libs;"
            "check_libs();"
            "print([m for m in ('torch','bs_roformer','torchaudio') if m in sys.modules])"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        assert proc.returncode == 0, f"子进程失败：{proc.stderr}"
        assert proc.stdout.strip() == "[]", f"L1 意外导入了重型模块：{proc.stdout.strip()}"


class TestL2ProbeDevice:
    @pytest.mark.gpu
    def test_returns_dict_with_ok_flag(self, env_probe):
        assert isinstance(env_probe, dict)
        assert "ok" in env_probe

    @pytest.mark.gpu
    def test_shape_contract(self, env_probe):
        """无论成功失败，键都必须齐——诊断面板会无差别地读它们。"""
        for key in ("ok", "backend", "device_name", "torch_version", "flash_attn"):
            assert key in env_probe, f"probe_device() 缺少 {key}，面板会因此崩"

    @pytest.mark.gpu
    def test_error_is_reported_as_string(self, env_probe):
        error = env_probe.get("error", "")
        assert isinstance(error, str), f"error 字段应为字符串，实际为 {type(error)}"

    def test_failure_is_not_fatal(self):
        """探测失败必须返回 ok=False 而不是抛异常。

        没装 torch 的机器上一启动就崩，正是这个工具最不该有的行为。
        """
        result = probe_device(timeout=30, force=True)
        assert result.get("ok") in (True, False)


class TestL3ReportRows:
    def test_row_schema(self):
        for row in build_report_rows():
            missing = {"item", "level", "detail", "remedy"} - set(row)
            assert not missing, f"{row.get('item', '?')} 缺字段：{missing}"
            assert row["level"] in LEVELS

    def test_row_count_matches_l1(self):
        """L3 只是 L1 的重新整形，两边条数必须一致，否则面板会漏项。"""
        assert len(build_report_rows()) == len(check_libs())


class TestDiagnosticText:
    def test_has_all_three_sections(self):
        """三个小节标题都必须出现。

        早先这里断言的是一个中文报告里根本不存在的英文标题 "Environment Status"，
        永远不会通过——测了个寂寞。
        """
        text = diagnostic_text()
        for header in ("【L1 静态库检测】", "【L2 动态设备探测】", "【路径诊断】"):
            assert header in text, f"诊断报告缺少小节 {header}"

    def test_includes_every_l1_row(self):
        text = diagnostic_text()
        for row in check_libs():
            assert row["item"] in text, f"诊断报告漏掉了 {row['item']}"

    def test_includes_path_probe_chain(self):
        """【路径诊断】必须有实际内容，不能是 ImportError 兜底文案。

        这里守的是 env_check 里那个真实的坑：paths 导入路径写错过
        （`core.paths` 而非 `audio_separation.paths`），异常被 except 吞掉，
        诊断报告看起来正常却从没打印过探测链。
        """
        text = diagnostic_text()
        tail = text.split("【路径诊断】", 1)[1]
        assert "模块不可用" not in tail, "【路径诊断】小节降级成了 ImportError 兜底文案"
        assert "路径诊断异常" not in tail
        assert tail.strip(), "【路径诊断】小节是空的"

    def test_prints_level_tags(self):
        text = diagnostic_text()
        assert "[PASS]" in text or "[FAIL]" in text or "[WARN]" in text or "[INFO]" in text
