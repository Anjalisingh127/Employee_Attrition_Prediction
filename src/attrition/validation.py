"""Command-line dataset validation."""

import json

from attrition.config import get_settings
from attrition.data import load_dataset, validate_dataset


def main() -> None:
    """Validate the configured dataset and print a reproducible JSON audit."""

    settings = get_settings()
    audit = validate_dataset(load_dataset(settings.dataset_path))
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
