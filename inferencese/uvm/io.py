"""Backing-store I/O for the unified virtual memory.

The SSD is the lowest tier: it always holds a complete copy of the virtual
address space.  Read-only segments are mapped *directly onto the model files*
(no second copy of the weights is ever written), writable segments live in a
sparse swap file.

Reads use O_DIRECT where the filesystem allows it so the kernel page cache does
not shadow our own RAM tier (which would silently double memory use), and fall
back to buffered reads plus ``posix_fadvise(DONTNEED)`` otherwise.
"""
from __future__ import annotations

import mmap
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ALIGN = 4096


def aligned_buffer(nbytes: int) -> np.ndarray:
    """Page-aligned, zero-initialised uint8 buffer (anonymous mmap)."""
    size = max(ALIGN, (nbytes + ALIGN - 1) // ALIGN * ALIGN)
    return np.frombuffer(mmap.mmap(-1, size), dtype=np.uint8)


class FileStore:
    """Positional reader over a set of files, addressed by small integer ids."""

    def __init__(self, direct: bool = True, drop_page_cache: bool = True,
                 io_threads: int = 8, max_request: int = 64 << 20):
        self.direct = direct
        self.drop_page_cache = drop_page_cache
        self.max_request = max_request
        self.paths: list[str] = []
        self._ids: dict[str, int] = {}
        self._fds: dict[int, tuple[int, bool]] = {}
        self._lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=max(1, io_threads),
                                       thread_name_prefix="isse-io")
        self.bytes_read = 0
        self.read_calls = 0

    # -- file table -------------------------------------------------------
    def add(self, path: str) -> int:
        path = os.path.abspath(path)
        with self._lock:
            if path not in self._ids:
                self._ids[path] = len(self.paths)
                self.paths.append(path)
            return self._ids[path]

    def _fd(self, fid: int) -> tuple[int, bool]:
        ent = self._fds.get(fid)
        if ent is None:
            with self._lock:
                ent = self._fds.get(fid)
                if ent is None:
                    path = self.paths[fid]
                    fd, direct = -1, False
                    if self.direct and hasattr(os, "O_DIRECT"):
                        try:
                            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                            direct = True
                        except OSError:
                            fd = -1
                    if fd < 0:
                        fd = os.open(path, os.O_RDONLY)
                    ent = self._fds[fid] = (fd, direct)
        return ent

    # -- reads ------------------------------------------------------------
    def read(self, fid: int, offset: int, nbytes: int) -> np.ndarray:
        """Read ``nbytes`` at ``offset``; returns a uint8 array view of
        exactly ``nbytes`` bytes backed by a page-aligned buffer."""
        fd, direct = self._fd(fid)
        start = offset - offset % ALIGN if direct else offset
        lead = offset - start
        span = lead + nbytes
        buf = aligned_buffer(span)
        want = (span + ALIGN - 1) // ALIGN * ALIGN if direct else span
        mv = memoryview(buf)
        done = 0
        while done < span:
            chunk = min(self.max_request, want - done)
            try:
                got = os.preadv(fd, [mv[done:done + chunk]], start + done)
            except OSError:
                if not direct:
                    raise
                # O_DIRECT refused at runtime (e.g. tmpfs/overlay): downgrade.
                os.close(fd)
                fd = os.open(self.paths[fid], os.O_RDONLY)
                direct = False
                self._fds[fid] = (fd, False)
                continue
            if got <= 0:
                break
            done += got
        if done < span:
            raise IOError(f"short read from {self.paths[fid]} @ {offset}: "
                          f"wanted {nbytes}, got {max(0, done - lead)}")
        if not direct and self.drop_page_cache and hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(fd, start, span, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        self.bytes_read += nbytes
        self.read_calls += 1
        return buf[lead:lead + nbytes]

    def submit(self, fid: int, offset: int, nbytes: int):
        return self.pool.submit(self.read, fid, offset, nbytes)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        for fd, _ in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()


class SwapFile:
    """Sparse swap file holding evicted *writable* blocks (model state, context
    caches).  Slots are allocated lazily on first write-back, so the file only
    consumes SSD space for state that actually left RAM."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        self.end = 0
        self._lock = threading.Lock()
        self.bytes_written = 0
        self.bytes_read = 0

    def alloc(self, nbytes: int) -> int:
        with self._lock:
            off = self.end
            self.end += (nbytes + ALIGN - 1) // ALIGN * ALIGN
            return off

    def write(self, offset: int, data: np.ndarray) -> None:
        mv = memoryview(np.ascontiguousarray(data)).cast("B")
        done = 0
        while done < len(mv):
            done += os.pwrite(self.fd, mv[done:], offset + done)
        self.bytes_written += len(mv)

    def read(self, offset: int, nbytes: int) -> np.ndarray:
        buf = aligned_buffer(nbytes)
        mv = memoryview(buf)
        done = 0
        while done < nbytes:
            got = os.preadv(self.fd, [mv[done:nbytes]], offset + done)
            if got <= 0:
                raise IOError("short read from swap file")
            done += got
        self.bytes_read += nbytes
        return buf[:nbytes]

    def close(self, unlink: bool = True):
        try:
            os.close(self.fd)
        except OSError:
            pass
        if unlink:
            try:
                os.unlink(self.path)
            except OSError:
                pass
