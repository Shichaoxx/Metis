"""Small explicit registries, with no import-time plugin discovery."""
from __future__ import annotations
import json
import os
from pathlib import Path
from .schema import load_manifest


class Registry:
    def __init__(self):
        self._items = {}

    def register(self, name, item=None):
        def add(value):
            if name in self._items:
                raise ValueError(f"Already registered: {name}")
            self._items[name] = value
            return value
        return add if item is None else add(item)

    def get(self, name):
        if name not in self._items:
            raise ValueError(f"Unknown registration {name!r}; available: {sorted(self._items)}")
        return self._items[name]

    def names(self):
        return sorted(self._items)


class DatasetRegistry:
    def __init__(self, path=".cometa/datasets.json"):
        self.path = Path(path).resolve()

    def entries(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def register(self, name, manifest_path, *, replace=False):
        path = Path(manifest_path).resolve()
        manifest = load_manifest(path)
        entries = self.entries()
        if name in entries and not replace:
            raise ValueError(f"Dataset {name!r} already registered; use --replace explicitly")
        entries[name] = {"path": os.path.relpath(path, self.path.parent),
                         "manifest_sha256": manifest["_manifest_sha256"]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
        return entries[name]

    def resolve(self, name):
        entries = self.entries()
        if name not in entries:
            raise ValueError(f"Dataset not registered: {name}")
        entry = entries[name]
        path = (self.path.parent / entry["path"]).resolve()
        manifest = load_manifest(path)
        if manifest["_manifest_sha256"] != entry["manifest_sha256"]:
            raise ValueError("Registered manifest changed; explicitly register new revision")
        return path

