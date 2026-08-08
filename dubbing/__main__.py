from . import nvlibs

nvlibs.preload()

from .cli import main  # noqa: E402

raise SystemExit(main())
