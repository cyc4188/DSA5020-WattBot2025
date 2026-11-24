"""Portability checks: provider config without OpenCode, and the Node agent.

These tests must not depend on the developer's real ``~/.config/opencode`` or
on a populated ``.env``; every case pins ``XDG_CONFIG_HOME`` to a temp dir and
reads API settings from a controlled environment only.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wattbot import multimodal
from wattbot.deepseek import CAMPUS_BASE

ROOT = Path(__file__).resolve().parents[1]
KEY_ENV = "HKUSTGZ_AIGC_API_KEY"
BASE_ENV = "HKUSTGZ_AIGC_BASE_URL"


def env_only_setting(name):
    """Stand-in for ``wattbot.deepseek.setting`` that ignores any repo ``.env``."""
    return os.environ.get(name)


class ProviderConfigTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.config_home = Path(self.tempdir.name)
        patcher = mock.patch.dict(
            os.environ,
            {k: v for k, v in os.environ.items() if k not in {KEY_ENV, BASE_ENV}},
            clear=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ["XDG_CONFIG_HOME"] = str(self.config_home)
        setting_patcher = mock.patch.object(
            multimodal, "setting", side_effect=env_only_setting)
        setting_patcher.start()
        self.addCleanup(setting_patcher.stop)

    def test_env_key_works_without_opencode_config(self):
        os.environ[KEY_ENV] = "test-key"
        self.assertFalse((self.config_home / "opencode" / "opencode.json").exists())
        base_url, api_key = multimodal.load_provider_config()
        self.assertEqual(base_url, CAMPUS_BASE)
        self.assertEqual(api_key, "test-key")

    def test_base_url_env_override(self):
        os.environ[KEY_ENV] = "test-key"
        os.environ[BASE_ENV] = "https://mirror.example.test/api/v2/"
        base_url, api_key = multimodal.load_provider_config()
        self.assertEqual(base_url, "https://mirror.example.test/api/v2")
        self.assertEqual(api_key, "test-key")

    def test_opencode_config_is_still_used_when_env_is_empty(self):
        options = {"baseURL": "https://config.example.test/api/v2/",
                   "apiKey": "config-key"}
        config = self.config_home / "opencode" / "opencode.json"
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps(
            {"provider": {"hkustgz-aigc": {"options": options}}}), encoding="utf-8")
        base_url, api_key = multimodal.load_provider_config()
        self.assertEqual(base_url, "https://config.example.test/api/v2")
        self.assertEqual(api_key, "config-key")

    def test_missing_key_and_config_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, KEY_ENV):
            multimodal.load_provider_config()

    def test_unknown_provider_without_config_still_fails(self):
        with self.assertRaisesRegex(ValueError, "No OpenCode configuration found"):
            multimodal.load_provider_config("some-other-provider")


class NodeAgentSyntaxTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node is not on PATH")
    def test_agent_v2_parses(self):
        script = ROOT / "pi_rag" / "agent_v2.mjs"
        self.assertTrue(script.is_file())
        result = subprocess.run(
            [shutil.which("node"), "--check", str(script)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
