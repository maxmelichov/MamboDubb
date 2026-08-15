"""`python -m dubbing` the pipeline's entry point.

Guarded by `__name__ == "__main__"` on purpose. Windows (and macOS since 3.8)
starts multiprocessing children with **spawn**, which re-imports this module
under the name `__mp_main__` to rebuild the child's namespace: an unguarded
`main()` here would start a second, argument-less run inside every worker
process torch or demucs ever creates. On POSIX-with-fork the guard costs
nothing, which is why it is unconditional rather than platform-tested.
"""

from . import nvlibs

nvlibs.preload()

from .cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
