#!/usr/bin/env python3
"""
Migration script to move delta_1 and delta_2 from player_args to game_args in bargaining game configs.

This script:
1. Finds all config.json files in the specified directory
2. For bargaining games, moves delta values from player_1_args/player_2_args to game_args
3. Backs up original files before modifying
4. Provides statistics on the migration

Usage:
    python migrate_delta_to_game_args.py [--data-dir DATA_DIR] [--dry-run] [--backup]
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple


def migrate_config(config: Dict) -> Tuple[Dict, bool]:
    """
    Migrate a single config by moving delta from player_args to game_args.

    Args:
        config: The configuration dictionary

    Returns:
        Tuple of (modified_config, was_modified)
    """
    # Only process bargaining games
    if config.get('game_type') != 'bargaining':
        return config, False

    modified = False

    # Extract delta values from player args
    player_1_args = config.get('player_1_args', {})
    player_2_args = config.get('player_2_args', {})

    delta_1 = player_1_args.get('delta')
    delta_2 = player_2_args.get('delta')

    # Check if delta values exist in player_args
    if delta_1 is None and delta_2 is None:
        return config, False

    # Ensure game_args exists
    if 'game_args' not in config:
        config['game_args'] = {}

    game_args = config['game_args']

    # Move delta_1 if it exists and isn't already in game_args
    if delta_1 is not None and 'delta_1' not in game_args:
        game_args['delta_1'] = delta_1
        del player_1_args['delta']
        modified = True

    # Move delta_2 if it exists and isn't already in game_args
    if delta_2 is not None and 'delta_2' not in game_args:
        game_args['delta_2'] = delta_2
        del player_2_args['delta']
        modified = True

    return config, modified


def find_config_files(data_dir: Path) -> list:
    """Find all config.json files in the data directory."""
    return list(data_dir.rglob('config.json'))


def migrate_file(file_path: Path, dry_run: bool, backup: bool) -> Tuple[bool, str]:
    """
    Migrate a single config file.

    Args:
        file_path: Path to the config file
        dry_run: If True, don't actually modify files
        backup: If True, create a backup of the original file

    Returns:
        Tuple of (was_modified, error_message)
    """
    try:
        # Read the config
        with open(file_path, 'r') as f:
            config = json.load(f)

        # Migrate the config
        modified_config, was_modified = migrate_config(config)

        if not was_modified:
            return False, ""

        if dry_run:
            return True, ""

        # Backup original file if requested
        if backup:
            backup_path = file_path.with_suffix('.json.bak')
            shutil.copy2(file_path, backup_path)

        # Write the modified config
        with open(file_path, 'w') as f:
            json.dump(modified_config, f, indent=4)

        return True, ""

    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Migrate delta values from player_args to game_args in bargaining game configs'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('GLEE/Data'),
        help='Path to the data directory (default: GLEE/Data)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without actually modifying files'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create .bak backups of original files before modifying'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information about each file processed'
    )

    args = parser.parse_args()

    # Validate data directory
    if not args.data_dir.exists():
        print(f"Error: Data directory '{args.data_dir}' does not exist")
        return 1

    print(f"Scanning for config files in: {args.data_dir}")

    # Find all config files
    config_files = find_config_files(args.data_dir)
    print(f"Found {len(config_files)} config.json files")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be modified ***\n")

    # Process each file
    total_modified = 0
    total_errors = 0
    bargaining_count = 0

    for file_path in config_files:
        was_modified, error = migrate_file(file_path, args.dry_run, args.backup)

        if error:
            total_errors += 1
            print(f"ERROR: {file_path}: {error}")
            continue

        if was_modified:
            total_modified += 1
            bargaining_count += 1
            if args.verbose:
                print(f"{'[DRY RUN] ' if args.dry_run else ''}Modified: {file_path}")

    # Print summary
    print("\n" + "="*60)
    print("MIGRATION SUMMARY")
    print("="*60)
    print(f"Total config files scanned: {len(config_files)}")
    print(f"Bargaining configs modified: {total_modified}")
    print(f"Errors encountered: {total_errors}")

    if args.dry_run:
        print("\nThis was a dry run. Re-run without --dry-run to apply changes.")
    elif args.backup:
        print(f"\nBackup files created with .bak extension")

    return 0 if total_errors == 0 else 1


if __name__ == '__main__':
    exit(main())
