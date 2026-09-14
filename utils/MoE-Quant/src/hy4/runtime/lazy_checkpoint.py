"""Defer payload verification until a process actually constructs model banks."""
from pathlib import Path
import threading


class LazyCheckpoint:
    def __init__(self, root, source_index_path, factory):
        if not (Path(root)/'hy4-checkpoint.index.json').is_file():
            raise ValueError('Checkpoint index missing')
        self._root = root
        self._source_index_path = source_index_path
        self._factory = factory
        self._value = None
        self._lock = threading.Lock()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        with self._lock:
            if self._value is None:
                self._value = self._factory(self._root, self._source_index_path)
        return getattr(self._value, name)
