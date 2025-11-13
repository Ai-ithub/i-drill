#!/usr/bin/env python3
"""
Update Dependencies Script for i-Drill Backend

This script helps update dependencies to latest secure versions.

Usage:
    python scripts/update_dependencies.py [--check] [--update] [--dry-run]
"""
import sys
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")

def print_error(text: str):
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def get_installed_versions() -> Dict[str, str]:
    """Get currently installed package versions"""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--format=json'],
            capture_output=True,
            text=True,
            check=True
        )
        import json
        packages = json.loads(result.stdout)
        return {pkg['name'].lower(): pkg['version'] for pkg in packages}
    except Exception as e:
        print_error(f"Error getting installed versions: {e}")
        return {}

def get_latest_versions(packages: List[str]) -> Dict[str, str]:
    """Get latest available versions for packages"""
    latest = {}
    for package in packages:
        try:
            # Remove version specifiers
            pkg_name = re.split(r'[<>=!]', package)[0].strip()
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'index', 'versions', pkg_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Parse version from output
                match = re.search(r'\((.+)\)', result.stdout)
                if match:
                    versions = match.group(1).split(', ')
                    if versions:
                        latest[pkg_name] = versions[0]  # Latest version
        except subprocess.TimeoutExpired:
            print_warning(f"Timeout checking {pkg_name}")
        except Exception as e:
            print_warning(f"Could not check {pkg_name}: {e}")
    
    return latest

def parse_requirements(file_path: Path) -> List[Tuple[str, str]]:
    """Parse requirements file and return (package, version) tuples"""
    requirements = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Skip -r includes
                if line.startswith('-r'):
                    continue
                
                # Parse package==version
                match = re.match(r'^([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)([<>=!]+)?(.+)?$', line)
                if match:
                    pkg = match.group(1)
                    op = match.group(2) or '=='
                    version = match.group(3) or ''
                    requirements.append((pkg, op + version if version else ''))
    except Exception as e:
        print_error(f"Error parsing requirements: {e}")
    
    return requirements

def check_outdated(requirements_file: Path) -> List[Dict]:
    """Check for outdated packages"""
    print_header("Checking for Outdated Packages")
    
    requirements = parse_requirements(requirements_file)
    installed = get_installed_versions()
    outdated = []
    
    for pkg, version_spec in requirements:
        # Remove extras [brackets]
        pkg_name = re.sub(r'\[.*\]', '', pkg).lower()
        
        if pkg_name in installed:
            current = installed[pkg_name]
            if version_spec:
                # Extract version from spec
                req_version = re.sub(r'[<>=!]+', '', version_spec)
                if current != req_version:
                    outdated.append({
                        'package': pkg_name,
                        'current': current,
                        'required': req_version,
                        'spec': pkg + version_spec
                    })
    
    return outdated

def update_requirements_file(requirements_file: Path, dry_run: bool = False):
    """Update requirements file with latest versions"""
    print_header("Updating Requirements File")
    
    if dry_run:
        print_warning("DRY RUN MODE - No files will be modified")
    
    requirements = parse_requirements(requirements_file)
    
    # Read original file
    with open(requirements_file, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    updated_content = original_content
    updates = []
    
    for pkg, version_spec in requirements:
        pkg_name = re.sub(r'\[.*\]', '', pkg).lower()
        
        try:
            # Get latest version using pip
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'index', 'versions', pkg_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Try to get latest version
                match = re.search(r'\((.+)\)', result.stdout)
                if match:
                    versions = match.group(1).split(', ')
                    if versions:
                        latest = versions[0]
                        old_line = f"{pkg}{version_spec}"
                        new_line = f"{pkg}=={latest}"
                        
                        if old_line in updated_content:
                            updated_content = updated_content.replace(old_line, new_line)
                            updates.append({
                                'package': pkg_name,
                                'old': old_line,
                                'new': new_line
                            })
        except Exception as e:
            print_warning(f"Could not update {pkg_name}: {e}")
    
    if updates:
        print_success(f"Found {len(updates)} packages to update:")
        for update in updates:
            print(f"  - {update['package']}: {update['old']} -> {update['new']}")
        
        if not dry_run:
            with open(requirements_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            print_success("Requirements file updated!")
        else:
            print_warning("Dry run - file not modified")
    else:
        print_success("All packages are up to date")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update dependencies')
    parser.add_argument('--check', action='store_true', help='Check for outdated packages')
    parser.add_argument('--update', action='store_true', help='Update requirements file')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no changes)')
    
    args = parser.parse_args()
    
    backend_path = Path(__file__).parent.parent
    requirements_file = backend_path.parent.parent / 'requirements' / 'backend.txt'
    
    if not requirements_file.exists():
        print_error(f"Requirements file not found: {requirements_file}")
        return 1
    
    if args.check:
        outdated = check_outdated(requirements_file)
        if outdated:
            print_warning(f"Found {len(outdated)} outdated packages:")
            for pkg in outdated:
                print(f"  - {pkg['package']}: {pkg['current']} (required: {pkg['required']})")
        else:
            print_success("All packages are up to date")
    
    if args.update:
        update_requirements_file(requirements_file, dry_run=args.dry_run)
    
    if not args.check and not args.update:
        print_header("Dependency Update Tool")
        print("Usage:")
        print("  python scripts/update_dependencies.py --check    # Check for outdated")
        print("  python scripts/update_dependencies.py --update    # Update requirements")
        print("  python scripts/update_dependencies.py --update --dry-run  # Preview changes")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

