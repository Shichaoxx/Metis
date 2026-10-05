"""Distribution checks use only synthetic archives, without models or extraction."""

import contextlib
import importlib.util
import io
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest
import zipfile


SPEC = importlib.util.spec_from_file_location(
    "check_distribution", Path(__file__).resolve().parents[1] / "scripts/check_distribution.py"
)
distribution = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(distribution)


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def wheel(self, extra=None, omit=()):
        path = self.root / "metis-0.1-py3-none-any.whl"
        files = {name: b"# module\n" for name in distribution.REQUIRED_MODULES if name not in omit}
        files.update(extra or {})
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        return path

    def sdist(self, extra=None, omit=()):
        path = self.root / "metis-0.1.tar.gz"
        files = {"src/" + name: b"# module\n" for name in distribution.REQUIRED_MODULES}
        files.update({"README.md": b"# Metis\n", "pyproject.toml": b"[project]\nname = 'metis'\n"})
        files.update(extra or {})
        with tarfile.open(path, "w:gz") as archive:
            for name, content in files.items():
                if name in omit:
                    continue
                member = tarfile.TarInfo("metis-0.1/" + name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        return path

    def test_public_docs_figures_and_artifact_module_are_allowed(self):
        for make, module in ((self.wheel, "metis/artifacts.py"), (self.sdist, "src/metis/artifacts.py")):
            with self.subTest(format=make.__name__):
                path = make({module: b"# artifact API\n",
                             "docs/architecture/figures/overview.pdf": b"%PDF-1.4\n\xff",
                             "docs/architecture/figures/overview.tex": b"diagram source"})
                self.assertEqual(distribution.check_distribution(path), [])

    def test_runtime_evidence_internal_documents_and_caches_are_rejected(self):
        private = ["data/sample.json", "runs/run.json", "reports/results.json", "artifacts/weights.bin",
                   "logs/train.log", "ops/supervisor.py", ".local/archive.py", ".git/config",
                   ".codex/config.toml", ".agents/state.json", "docs/operations/handoff.md",
                   "docs/history/run.md", "AGENTS.md", "CORE.md", "HANDOFF.md", "JUPITER.md",
                   ".env", ".env.production", "metis/__pycache__/module.pyc",
                   ".pytest_cache/README.md", "docs/architecture/figures/overview.aux"]
        for make in (self.wheel, self.sdist):
            for name in private:
                with self.subTest(format=make.__name__, member=name):
                    errors = distribution.check_distribution(make({name: b"local-only"}))
                    self.assertTrue(any(name in error and "local-only" in error for error in errors), errors)

    def test_required_task_modules_and_sdist_metadata_are_checked(self):
        for module in distribution.REQUIRED_MODULES:
            self.assertTrue(any("missing required" in e and module in e
                                for e in distribution.check_distribution(self.wheel(omit=(module,)))))
        for module in ("src/metis/tasks/objectives.py", "src/metis/tasks/decisions.py", "README.md", "pyproject.toml"):
            self.assertTrue(any("missing required" in e and module in e
                                for e in distribution.check_distribution(self.sdist(omit=(module,)))))

    def test_private_literal_report_hides_values_and_keeps_pattern_order(self):
        first, second = "synthetic-contact-token", "synthetic-machine-token"
        path = self.sdist({"docs/example.md": (second + " " + first).encode(),
                           "docs/binary.png": b"\xff\xfe"})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = distribution.main([str(path), "--forbid-text", first, "--forbid-text", second])
        self.assertEqual(code, 1)
        self.assertIn("docs/example.md", output.getvalue())
        self.assertIn("pattern #1", output.getvalue())
        self.assertIn("pattern #2", output.getvalue())
        self.assertNotIn(first, output.getvalue())
        self.assertNotIn(second, output.getvalue())
        self.assertNotIn("binary.png", output.getvalue())

    def test_absolute_traversal_and_windows_paths_are_rejected_without_extraction(self):
        for name in ("/outside.py", "../outside.py", "metis/../outside.py", "C:/outside.py", "metis\\outside.py"):
            with self.subTest(member=name):
                path = self.wheel({name: b"not extracted"})
                self.assertTrue(any("unsafe archive path" in e for e in distribution.check_distribution(path)))
        path = self.sdist({"../outside.py": b"not extracted"})
        self.assertTrue(any("unsafe archive path" in e for e in distribution.check_distribution(path)))
        self.assertFalse((self.root / "outside.py").exists())

    def test_symlinks_and_tar_hardlinks_are_rejected(self):
        path = self.wheel()
        with zipfile.ZipFile(path, "a") as archive:
            member = zipfile.ZipInfo("metis/linked.py")
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(member, "__init__.py")
        self.assertTrue(any("links and special" in e for e in distribution.check_distribution(path)))
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            path = self.root / "linked.tar"
            with tarfile.open(path, "w") as archive:
                member = tarfile.TarInfo("metis-0.1/src/metis/linked.py")
                member.type = kind
                member.linkname = "__init__.py"
                archive.addfile(member)
            self.assertTrue(any("links and special" in e for e in distribution.check_distribution(path)))

    def test_multiple_sdist_roots_and_malformed_archive_fail(self):
        path = self.root / "roots.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("first/README.md", "readme")
            archive.writestr("second/pyproject.toml", "[project]")
        self.assertIn("source distribution requires exactly one root directory",
                      distribution.check_distribution(path))
        path = self.root / "invalid.whl"
        path.write_bytes(b"not a zip archive")
        self.assertIn("cannot read distribution archive", distribution.check_distribution(path))


if __name__ == "__main__":
    unittest.main()
