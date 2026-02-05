#!/usr/bin/env python3
"""
Download KFall and SisFall datasets for fall detection research.

Uses Kaggle API for reliable downloads. Requires Kaggle API credentials.

Setup:
    1. Create Kaggle account: https://www.kaggle.com
    2. Get API token: https://www.kaggle.com/settings → API → Create New Token
    3. Place kaggle.json in ~/.kaggle/kaggle.json
    4. Install: uv add kagglehub

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --force
    python scripts/download_datasets.py --sisfall-only
"""

import argparse
import shutil
from pathlib import Path

try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("⚠️  kagglehub not installed. Install with: uv add kagglehub")


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_json.exists():
        print("\n❌ Kaggle credentials not found!")
        print("\nSetup instructions:")
        print("  1. Create Kaggle account: https://www.kaggle.com")
        print("  2. Go to: https://www.kaggle.com/settings")
        print("  3. Click 'Create New Token' under API section")
        print("  4. Place downloaded kaggle.json in: ~/.kaggle/")
        print("  5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False

    return True


def download_sisfall_kaggle(data_dir: Path, force: bool = False) -> None:
    """Download SisFall dataset from Kaggle."""
    print("\n" + "="*80)
    print("DOWNLOADING SISFALL DATASET (via Kaggle)")
    print("="*80)

    sisfall_target = data_dir / "SisFall"

    # Check if already exists
    if sisfall_target.exists() and not force:
        num_files = len(list(sisfall_target.rglob("*.txt")))
        print(f"⚠️  SisFall already exists: {sisfall_target}")
        print(f"   Contains {num_files} files")
        response = input("Delete and re-download? [y/N]: ").strip().lower()
        if response != 'y':
            print("✓ Skipping SisFall download.")
            return
        print("🗑️  Removing existing SisFall directory...")
        shutil.rmtree(sisfall_target)

    # Create target directory
    sisfall_target.mkdir(parents=True, exist_ok=True)

    # Download from Kaggle
    print(f"\n📥 Downloading SisFall from Kaggle...")
    print(f"   Dataset: nvnikhil0001/sis-fall-original-dataset")
    print(f"   Target: {sisfall_target}")

    try:
        download_path = kagglehub.dataset_download("nvnikhil0001/sis-fall-original-dataset")
        download_path = Path(download_path)
        print(f"\n✅ Downloaded to cache: {download_path}")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("  - Check Kaggle credentials: ~/.kaggle/kaggle.json")
        print("  - Verify internet connection")
        print("  - Accept dataset terms: https://www.kaggle.com/datasets/nvnikhil0001/sis-fall-original-dataset")
        return

    # Copy files to organized location
    print(f"\n📦 Organizing files to {sisfall_target}...")

    file_count = 0
    for item in download_path.rglob('*'):
        if item.is_file():
            relative_path = item.relative_to(download_path)
            dest = sisfall_target / relative_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)
            file_count += 1
            if file_count % 100 == 0:
                print(f"   Copied {file_count} files...", end='\r')

    print(f"   Copied {file_count} files" + " "*20)

    # Verify
    verify_sisfall(sisfall_target)


def download_kfall_kaggle(data_dir: Path, force: bool = False) -> None:
    """Download KFall dataset from Kaggle."""
    print("\n" + "="*80)
    print("DOWNLOADING KFALL DATASET (via Kaggle)")
    print("="*80)

    kfall_target = data_dir / "KFall"

    # Check if already exists
    if kfall_target.exists() and not force:
        sensor_dir = kfall_target / "sensor_data"
        if sensor_dir.exists():
            num_files = len(list(sensor_dir.rglob("*.csv")))
            print(f"⚠️  KFall already exists: {kfall_target}")
            print(f"   Contains {num_files} files")
            response = input("Delete and re-download? [y/N]: ").strip().lower()
            if response != 'y':
                print("✓ Skipping KFall download.")
                return
        print("🗑️  Removing existing KFall directory...")
        shutil.rmtree(kfall_target)

    # Create target directory
    kfall_target.mkdir(parents=True, exist_ok=True)

    # Download from Kaggle
    print(f"\n📥 Downloading KFall from Kaggle...")
    print(f"   Dataset: usmanabbasi2002/kfall-dataset")
    print(f"   Target: {kfall_target}")

    try:
        download_path = kagglehub.dataset_download("usmanabbasi2002/kfall-dataset")
        download_path = Path(download_path)
        print(f"\n✅ Downloaded to cache: {download_path}")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("  - Check Kaggle credentials: ~/.kaggle/kaggle.json")
        print("  - Verify internet connection")
        print("  - Accept dataset terms: https://www.kaggle.com/datasets/usmanabbasi2002/kfall-dataset")
        return

    # Copy files to organized location
    print(f"\n📦 Organizing files to {kfall_target}...")

    file_count = 0
    for item in download_path.rglob('*'):
        if item.is_file():
            relative_path = item.relative_to(download_path)
            dest = kfall_target / relative_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)
            file_count += 1
            if file_count % 100 == 0:
                print(f"   Copied {file_count} files...", end='\r')

    print(f"   Copied {file_count} files" + " "*20)

    # Verify
    verify_kfall(kfall_target)


def verify_sisfall(sisfall_dir: Path) -> None:
    """Verify SisFall dataset integrity."""
    print(f"\n🔍 Verifying SisFall dataset...")

    if not sisfall_dir.exists():
        print(f"❌ SisFall directory not found: {sisfall_dir}")
        return

    # Count subjects (SA* and SE* directories)
    subjects = list(sisfall_dir.glob("SA*")) + list(sisfall_dir.glob("SE*"))
    num_subjects = len(subjects)

    # Count data files
    txt_files = list(sisfall_dir.rglob("*.txt"))
    num_files = len(txt_files)

    print(f"   Subjects: {num_subjects}")
    print(f"   Data files: {num_files}")

    # Expected values
    if num_subjects >= 38 and num_files >= 4000:
        print(f"✅ SisFall dataset verified!")
    elif num_subjects > 0 or num_files > 0:
        print(f"⚠️  SisFall downloaded but counts differ from expected")
        print(f"   Expected: ~38 subjects, ~4500 files")
        print(f"   This may be normal depending on Kaggle version")
    else:
        print(f"❌ SisFall appears empty")


def verify_kfall(kfall_dir: Path) -> None:
    """Verify KFall dataset integrity."""
    print(f"\n🔍 Verifying KFall dataset...")

    if not kfall_dir.exists():
        print(f"❌ KFall directory not found: {kfall_dir}")
        return

    # Check for sensor_data directory (might not exist in Kaggle version)
    sensor_dir = kfall_dir / "sensor_data"
    label_dir = kfall_dir / "label_data"

    # Count CSV files (could be in root or sensor_data/)
    csv_files = list(kfall_dir.rglob("*.csv"))
    num_files = len(csv_files)

    # Count subject directories
    subjects = list(kfall_dir.glob("S*"))
    if sensor_dir.exists():
        subjects = list(sensor_dir.glob("S*"))
    num_subjects = len(subjects)

    print(f"   Subjects: {num_subjects}")
    print(f"   CSV files: {num_files}")

    if sensor_dir.exists():
        print(f"   Structure: sensor_data/ directory found")
    if label_dir.exists():
        print(f"   Structure: label_data/ directory found")

    # Expected values (flexible for different structures)
    if num_files >= 5000:
        print(f"✅ KFall dataset verified!")
    elif num_files > 0:
        print(f"⚠️  KFall downloaded but file count differs from expected")
        print(f"   Expected: ~5000+ files")
        print(f"   This may be normal depending on Kaggle version")
    else:
        print(f"❌ KFall appears empty")


def print_summary(data_dir: Path) -> None:
    """Print summary of downloaded datasets."""
    print("\n" + "="*80)
    print("DATASET DOWNLOAD SUMMARY")
    print("="*80)

    sisfall_dir = data_dir / "SisFall"
    kfall_dir = data_dir / "KFall"

    print(f"\n📁 Data directory: {data_dir.absolute()}")
    print()

    # SisFall status
    if sisfall_dir.exists():
        subjects = len(list(sisfall_dir.glob("SA*")) + list(sisfall_dir.glob("SE*")))
        files = len(list(sisfall_dir.rglob("*.txt")))
        size_mb = sum(f.stat().st_size for f in sisfall_dir.rglob("*") if f.is_file()) / (1024*1024)
        print(f"   ✅ SisFall")
        print(f"      Location: {sisfall_dir}")
        print(f"      Subjects: {subjects}")
        print(f"      Files: {files}")
        print(f"      Size: {size_mb:.1f} MB")
    else:
        print(f"   ❌ SisFall: Not downloaded")

    print()

    # KFall status
    if kfall_dir.exists():
        csv_files = list(kfall_dir.rglob("*.csv"))
        num_files = len(csv_files)
        size_mb = sum(f.stat().st_size for f in kfall_dir.rglob("*") if f.is_file()) / (1024*1024)

        sensor_dir = kfall_dir / "sensor_data"
        subjects = len(list(sensor_dir.glob("S*"))) if sensor_dir.exists() else len(list(kfall_dir.glob("S*")))

        print(f"   ✅ KFall")
        print(f"      Location: {kfall_dir}")
        print(f"      Subjects: {subjects}")
        print(f"      Files: {num_files}")
        print(f"      Size: {size_mb:.1f} MB")
    else:
        print(f"   ❌ KFall: Not downloaded")

    print("\n" + "="*80)
    print("✅ DOWNLOAD COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Explore data: jupyter notebook notebooks/")
    print("  2. Run preprocessing: python scripts/preprocess_data.py")
    print("\nDataset sources:")
    print("  - SisFall: https://www.kaggle.com/datasets/nvnikhil0001/sis-fall-original-dataset")
    print("  - KFall: https://www.kaggle.com/datasets/usmanabbasi2002/kfall-dataset")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download KFall and SisFall datasets using Kaggle API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download both datasets:
    python scripts/download_datasets.py

  Force re-download:
    python scripts/download_datasets.py --force

  Download only SisFall:
    python scripts/download_datasets.py --sisfall-only

  Custom data directory:
    python scripts/download_datasets.py --data-dir data/raw

Setup Kaggle API:
  1. Get token: https://www.kaggle.com/settings (API section)
  2. Place kaggle.json in ~/.kaggle/
  3. Run: chmod 600 ~/.kaggle/kaggle.json
  4. Accept dataset terms on Kaggle website

Datasets:
  - SisFall: https://www.kaggle.com/datasets/nvnikhil0001/sis-fall-original-dataset
  - KFall: https://www.kaggle.com/datasets/usmanabbasi2002/kfall-dataset
        """
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("fall_detection_data"),
        help="Directory to store datasets (default: fall_detection_data/)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if datasets exist"
    )
    parser.add_argument(
        "--sisfall-only",
        action="store_true",
        help="Download only SisFall dataset"
    )
    parser.add_argument(
        "--kfall-only",
        action="store_true",
        help="Download only KFall dataset"
    )

    args = parser.parse_args()

    # Check dependencies
    if not KAGGLE_AVAILABLE:
        print("❌ kagglehub not installed")
        print("\nInstall with: uv add kagglehub")
        return 1

    if not check_kaggle_credentials():
        return 1

    # Create data directory
    args.data_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("FALL DETECTION DATASET DOWNLOADER")
    print("Downloading from Kaggle")
    print("="*80)
    print(f"\nData directory: {args.data_dir.absolute()}")

    # Download datasets
    if args.kfall_only:
        download_kfall_kaggle(args.data_dir, args.force)
    elif args.sisfall_only:
        download_sisfall_kaggle(args.data_dir, args.force)
    else:
        download_sisfall_kaggle(args.data_dir, args.force)
        download_kfall_kaggle(args.data_dir, args.force)

    # Print summary
    print_summary(args.data_dir)

    return 0


if __name__ == "__main__":
    exit(main())
