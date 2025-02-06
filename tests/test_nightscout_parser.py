import pytest
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import sys
import json
import os
from glupredkit.helpers.cli import read_data_from_csv

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glupredkit.parsers.nightscout import Parser

# Setup fixtures
@pytest.fixture(scope="module")
def test_dir():
    """Fixture for the test data directory."""
    return Path(__file__).parent / "example_data"

@pytest.fixture(scope="module")
def date_range():
    """Fixture for the date range used in tests."""
    start_date = datetime(2024, 10, 26, 0, 0, tzinfo=timezone.utc)
    end_date = datetime(2024, 10, 28, 0, 0, tzinfo=timezone.utc)
    return start_date, end_date

def test_parser(test_dir, date_range):
    """Test the parser with our test data files."""
    start_date, end_date = date_range

    # Initialize parser
    parser = Parser()

    print(f"Test directory: {test_dir}")
    print(f"Looking for files:")
    print(f"- {test_dir / 'nightscout_profiles.json'}")
    print(f"- {test_dir / 'nightscout_treatments.json'}")
    print(f"- {test_dir / 'nightscout_entries.json'}")

    # Verify test files exist
    assert (test_dir / "nightscout_profiles.json").exists(), "Profiles file not found"
    assert (test_dir / "nightscout_treatments.json").exists(), "Treatments file not found"
    assert (test_dir / "nightscout_entries.json").exists(), "Entries file not found"

    # Load test data from local files
    with open(test_dir / "nightscout_profiles.json") as f:
        profiles = json.load(f)
    with open(test_dir / "nightscout_treatments.json") as f:
        treatments_data = json.load(f)
    treatments = []
    for t in treatments_data:
        treatment = type('Treatment', (), {})()
        for k, v in t.items():
            setattr(treatment, k, v)
        treatments.append(treatment)
    with open(test_dir / "nightscout_entries.json") as f:
        entries_data = json.load(f)
    entries = []
    for e in entries_data:
        entry = type('Entry', (), {})()
        for k, v in e.items():
            setattr(entry, k, v)
        entries.append(entry)

    df = parser.process_data(entries, treatments, profiles, start_date, end_date)

    # Save output to a snapshot file if it doesn't exist
    snapshot_file_name = "nightscout_expected_output.csv"
    snapshot_file = test_dir / snapshot_file_name
    if not snapshot_file.exists():
        df.to_csv(snapshot_file)
        print(f"\nCreated initial snapshot at {snapshot_file}")
        print("Please verify the output manually before proceeding with tests")
        return

    # Load expected output in the same way as in cli helpers
    expected_df = read_data_from_csv(test_dir, snapshot_file_name)

    # Remove timezone info from the index, because the nighscout parser sets the computers time zone in the df
    df.index = df.index.tz_convert("UTC")
    expected_df.index = expected_df.index.tz_convert("UTC")

    # Ensure same type of int for column hour
    df['hour'] = df['hour'].astype('int64')
    expected_df['hour'] = expected_df['hour'].astype('int64')

    # Compare output with expected
    pd.testing.assert_frame_equal(df, expected_df)

