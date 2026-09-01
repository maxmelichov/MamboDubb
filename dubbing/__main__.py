"""`python -m dubbing` the pipeline's entry point.

Guarded by `__name__ == "__main__"` on purpose. Windows (and macOS since 3.8)
starts multiprocessing children with **spawn**, which re-imports this module
under the name `__mp_main__` to rebuild the child's namespace: an unguarded
`main()` here would start a second, argument-less run inside every worker
process torch or demucs ever creates. On POSIX-with-fork the guard costs
nothing, which is why it is unconditional rather than platform-tested.
"""

from . import nvlibs

# Outside the guard, deliberately, and the only thing here that is. Registering
# the wheels' CUDA directories has to happen in *every* process before the first
# extension module is loaded, and a spawned child is a process: it re-imports
# this module as `__mp_main__` and then loads torch, so a `preload()` inside the
# guard would leave every demucs worker on Windows unable to find cuBLAS.
# Calling it twice is free; it is idempotent by construction.
nvlibs.preload()

from .cli import main  # noqa: E402

if __name__ == "__main__":
    # Inside the guard, and it did not used to be. Spawn re-imports this module
    # for every worker torch and demucs create, so a warning at module scope was
    # printed once per child: the one line whose whole value is being the first
    # thing in the log arrived eight more times, in the middle of the stem
    # separation it was warning about. `once=True` suppresses repeats within a
    # process and cannot see across one.
    #
    # Before the first stage, not after the slow one. A CPU-only torch on a
    # machine with a card is not an error anywhere in the pipeline, so nothing
    # raises and nothing else would ever say it out loud.
    nvlibs.warn_if_gpu_unused()
    raise SystemExit(main())
