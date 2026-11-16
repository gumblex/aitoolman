import argparse
import contextlib
import io
import unittest
from unittest import mock

from aitoolman import cli


def _build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser_name", required=True)
    cli.subparser_client(subparsers)
    cli.subparser_code_edit(subparsers)
    cli.subparser_version(subparsers)
    return parser


class TestVersionSubcommand(unittest.TestCase):
    def test_version_registered(self):
        parser = _build_parser()
        args = parser.parse_args(["version"])
        self.assertEqual(args.subparser_name, "version")

    def test_run_version_prints_version(self):
        from aitoolman import VERSION
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.run_version(mock.Mock())
        self.assertEqual(out.getvalue().strip(), f"aitoolman {VERSION}")

    def test_main_version(self):
        from aitoolman import VERSION
        out = io.StringIO()
        with mock.patch("sys.argv", ["aitoolman", "version"]):
            with contextlib.redirect_stdout(out):
                cli.main()
        self.assertEqual(out.getvalue().strip(), f"aitoolman {VERSION}")


class TestSingleModelOption(unittest.TestCase):
    def test_client_model_single_value(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["client", "-c", "config.toml", "-m", "fast"])
        self.assertEqual(args.model, "fast")

    def test_client_model_accepts_comma_string(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["client", "-c", "config.toml", "-m", "code,fast:2"])
        self.assertEqual(args.model, "code,fast:2")

    def test_code_edit_model_single_value(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["code-edit", "-c", "config.toml", "-m", "code,fast:2"])
        self.assertEqual(args.model, "code,fast:2")


if __name__ == '__main__':
    unittest.main()
