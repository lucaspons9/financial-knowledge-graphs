"""
OpenAI Language Model and Batch Processing functionality.
"""

# Batch processing configuration constants
# Default directory for storing batch data
DEFAULT_BATCH_DIR = "data/batch_processing"

# Default batch size
DEFAULT_BATCH_SIZE = 2000

# Default wait interval in seconds for status checks
DEFAULT_WAIT_INTERVAL = 30

# Maximum completion window in hours
COMPLETION_WINDOW = "24h"

# Batch folder name pattern (simplified)
BATCH_FOLDER_PATTERN = r'^batch_\d+$'

# Execution directory prefix
EXECUTION_PREFIX = "execution_" 