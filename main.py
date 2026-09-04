#!/usr/bin/env python3
"""Main entrypoint for OmniAgent."""

import os
import warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=Warning)
warnings.simplefilter("ignore")

if __name__ == "__main__":
    from omni_agent.cli import main
    main()
