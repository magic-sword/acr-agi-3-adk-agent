from pathlib import Path
import unittest

import yaml

from agent.model_runtime import SPECULATIVE_ARGS

ROOT = Path(__file__).resolve().parents[1]


class ModelServerArgsTests(unittest.TestCase):
    def test_local_server_uses_the_submission_speculative_decoding(self):
        command = [str(a) for a in yaml.safe_load((ROOT/'compose.yaml').read_text())['services']['vlm']['command']]
        start = command.index(SPECULATIVE_ARGS[0])
        self.assertEqual(command[start:start+len(SPECULATIVE_ARGS)], SPECULATIVE_ARGS)


if __name__ == '__main__':
    unittest.main()
