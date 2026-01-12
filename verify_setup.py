#!/usr/bin/env python
"""
Verification script - Check if project is properly set up
"""

import sys
from pathlib import Path

def check_requirements():
    """Check if requirements.txt exists"""
    if Path("requirements.txt").exists():
        print("✓ requirements.txt found")
        return True
    print("✗ requirements.txt NOT found")
    return False

def check_main_script():
    """Check if main.py exists"""
    if Path("main.py").exists():
        print("✓ main.py found")
        return True
    print("✗ main.py NOT found")
    return False

def check_config():
    """Check if config module exists"""
    if Path("config/config.py").exists():
        print("✓ config/config.py found")
        return True
    print("✗ config/config.py NOT found")
    return False

def check_data_module():
    """Check if data module exists"""
    if Path("data/dataset_loader.py").exists():
        print("✓ data/dataset_loader.py found")
        return True
    print("✗ data/dataset_loader.py NOT found")
    return False

def check_utils():
    """Check if utils module exists"""
    if Path("utils/dataset_utils.py").exists():
        print("✓ utils/dataset_utils.py found")
        return True
    print("✗ utils/dataset_utils.py NOT found")
    return False

def check_documentation():
    """Check if documentation files exist"""
    docs = [
        "QUICKSTART.md",
        "DATASET_README.md",
        "NOTEBOOK_EXAMPLES.md",
        "PROJECT_STRUCTURE.md",
        "SETUP_SUMMARY.md"
    ]
    
    all_exist = True
    for doc in docs:
        if Path(doc).exists():
            print(f"✓ {doc} found")
        else:
            print(f"✗ {doc} NOT found")
            all_exist = False
    
    return all_exist

def check_gitignore():
    """Check if .gitignore exists"""
    if Path(".gitignore").exists():
        print("✓ .gitignore found")
        return True
    print("✗ .gitignore NOT found")
    return False

def check_directories():
    """Check if required directories are created or creatable"""
    dirs = [
        "config",
        "data",
        "models",
        "utils"
    ]
    
    all_exist = True
    for d in dirs:
        if Path(d).exists():
            print(f"✓ {d}/ directory exists")
        else:
            print(f"✗ {d}/ directory NOT found")
            all_exist = False
    
    return all_exist

def check_imports():
    """Check if imports work"""
    try:
        from config.config import DATASET_CONFIG
        print("✓ config imports work")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("PROJECT SETUP VERIFICATION")
    print("="*60 + "\n")
    
    print("📁 CHECKING DIRECTORIES...")
    print("-" * 60)
    dir_ok = check_directories()
    print()
    
    print("📄 CHECKING CORE FILES...")
    print("-" * 60)
    req_ok = check_requirements()
    main_ok = check_main_script()
    config_ok = check_config()
    data_ok = check_data_module()
    utils_ok = check_utils()
    print()
    
    print("📚 CHECKING DOCUMENTATION...")
    print("-" * 60)
    docs_ok = check_documentation()
    print()
    
    print("🔧 CHECKING CONFIGURATION...")
    print("-" * 60)
    git_ok = check_gitignore()
    import_ok = check_imports()
    print()
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    all_checks = [dir_ok, req_ok, main_ok, config_ok, data_ok, utils_ok, docs_ok, git_ok]
    passed = sum(all_checks)
    total = len(all_checks)
    
    print(f"\nChecks Passed: {passed}/{total}")
    
    if all(all_checks):
        print("\n✅ ALL CHECKS PASSED - Project is ready!")
        print("\nNext steps:")
        print("1. pip install -r requirements.txt")
        print("2. python main.py --load")
        print("\nRead QUICKSTART.md for more options.")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\nPlease verify the file structure and try again.")
        print("Read PROJECT_STRUCTURE.md for the expected structure.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
