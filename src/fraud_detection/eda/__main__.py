"""Allow ``python -m fraud_detection.eda`` (the DOC-05 M2 command)."""

import sys

from fraud_detection.eda.cli import main

sys.exit(main())
