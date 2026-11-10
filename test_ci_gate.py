import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from run_ci_gate import execute

class CIGateTests(unittest.TestCase):
    def test_healthy_candidate_passes(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(execute('lexical',Path(tmp)),0)
    def test_empty_candidate_fails_gate(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertEqual(execute('empty',Path(tmp)),1)
            self.assertIn('REGRESSION:',log.getvalue())
    def test_unknown_mode_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):execute('typo',Path(tmp))

if __name__=='__main__':unittest.main()
