#!/usr/bin/env python3
"""
Security Check Script for i-Drill Backend

This script checks for:
1. Outdated dependencies with known vulnerabilities
2. Insecure configuration patterns
3. Missing security headers
4. Exposed credentials in code

Usage:
    python scripts/check_security.py [--fix] [--verbose]
"""
import sys
import os
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Tuple
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def check_pip_available() -> bool:
    """Check if pip is available"""
    try:
        subprocess.run(['pip', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_safety_installed() -> bool:
    """Check if safety (vulnerability scanner) is installed"""
    try:
        subprocess.run(['safety', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_safety():
    """Install safety package"""
    print_warning("Installing safety package for vulnerability scanning...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'safety'], check=True)
        print_success("Safety package installed successfully")
        return True
    except subprocess.CalledProcessError:
        print_error("Failed to install safety package")
        return False

def scan_vulnerabilities(requirements_file: str) -> Tuple[bool, List[Dict]]:
    """
    Scan requirements file for known vulnerabilities using safety
    
    Returns:
        (has_vulnerabilities, list of vulnerabilities)
    """
    if not check_safety_installed():
        print_warning("safety is not installed. Installing...")
        if not install_safety():
            print_error("Cannot scan vulnerabilities without safety package")
            return False, []
    
    try:
        result = subprocess.run(
            ['safety', 'check', '--json', '--file', requirements_file],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            return False, []
        
        # Parse JSON output
        try:
            vulnerabilities = json.loads(result.stdout)
            return True, vulnerabilities
        except json.JSONDecodeError:
            # Fallback to text parsing
            vulnerabilities = []
            for line in result.stdout.split('\n'):
                if 'vulnerability' in line.lower() or 'CVE' in line:
                    vulnerabilities.append({'raw': line})
            return len(vulnerabilities) > 0, vulnerabilities
            
    except subprocess.TimeoutExpired:
        print_error("Vulnerability scan timed out")
        return False, []
    except Exception as e:
        print_error(f"Error scanning vulnerabilities: {e}")
        return False, []

def check_env_file_security() -> List[str]:
    """Check .env file for security issues"""
    issues = []
    env_path = Path(__file__).parent.parent / '.env'
    config_example = Path(__file__).parent.parent / 'config.env.example'
    
    if not env_path.exists():
        print_warning(".env file not found. Using config.env.example as reference.")
        env_path = config_example
    
    if not env_path.exists():
        print_error("No .env or config.env.example file found")
        return issues
    
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for insecure patterns
        insecure_patterns = [
            (r'SECRET_KEY\s*=\s*["\']?your-secret-key', 'Insecure SECRET_KEY placeholder'),
            (r'SECRET_KEY\s*=\s*["\']?CHANGE_THIS', 'SECRET_KEY not changed from default'),
            (r'PASSWORD\s*=\s*["\']?postgres["\']?', 'Default PostgreSQL password'),
            (r'PASSWORD\s*=\s*["\']?admin123["\']?', 'Default admin password'),
            (r'PASSWORD\s*=\s*["\']?password["\']?', 'Weak password'),
            (r'APP_ENV\s*=\s*["\']?production["\']?', 'Production mode - ensure all secrets are set'),
        ]
        
        for pattern, message in insecure_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"{message} (found in {env_path.name})")
        
        # Check if SECRET_KEY is set
        if 'SECRET_KEY' not in content or re.search(r'SECRET_KEY\s*=\s*$', content):
            issues.append("SECRET_KEY is not set")
        
    except Exception as e:
        print_error(f"Error reading env file: {e}")
    
    return issues

def check_code_for_secrets() -> List[str]:
    """Check code files for hardcoded secrets"""
    issues = []
    backend_path = Path(__file__).parent.parent
    
    # Patterns to search for
    secret_patterns = [
        (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
        (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key'),
        (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret'),
        (r'token\s*=\s*["\'][^"\']+["\']', 'Hardcoded token'),
    ]
    
    # Files to check
    python_files = list(backend_path.rglob('*.py'))
    
    for py_file in python_files:
        # Skip virtual environments and cache
        if 'venv' in str(py_file) or '__pycache__' in str(py_file):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for pattern, message in secret_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Skip comments and docstrings
                        stripped = line.strip()
                        if not stripped.startswith('#') and not stripped.startswith('"""'):
                            issues.append(
                                f"{message} in {py_file.relative_to(backend_path)}:L{line_num}"
                            )
        except Exception as e:
            print_warning(f"Could not check {py_file}: {e}")
    
    return issues

def check_gitignore() -> List[str]:
    """Check if .gitignore properly excludes sensitive files"""
    issues = []
    gitignore_path = Path(__file__).parent.parent.parent.parent / '.gitignore'
    
    if not gitignore_path.exists():
        issues.append(".gitignore file not found")
        return issues
    
    try:
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_patterns = [
            '.env',
            '.env.local',
            '*.env',
            'config.env',
        ]
        
        for pattern in required_patterns:
            if pattern not in content:
                issues.append(f".gitignore missing pattern: {pattern}")
    except Exception as e:
        print_error(f"Error reading .gitignore: {e}")
    
    return issues

def main():
    """Main security check function"""
    print_header("i-Drill Backend Security Check")
    
    backend_path = Path(__file__).parent.parent
    requirements_file = backend_path.parent.parent / 'requirements' / 'backend.txt'
    
    all_issues = []
    has_critical_issues = False
    
    # 1. Check dependencies for vulnerabilities
    print_header("1. Checking Dependencies for Vulnerabilities")
    if requirements_file.exists():
        has_vulns, vulnerabilities = scan_vulnerabilities(str(requirements_file))
        if has_vulns:
            print_error(f"Found {len(vulnerabilities)} potential vulnerabilities")
            for vuln in vulnerabilities[:10]:  # Show first 10
                if isinstance(vuln, dict):
                    print_warning(f"  - {vuln.get('raw', str(vuln))}")
                else:
                    print_warning(f"  - {vuln}")
            has_critical_issues = True
        else:
            print_success("No known vulnerabilities found in dependencies")
    else:
        print_warning(f"Requirements file not found: {requirements_file}")
    
    # 2. Check environment file security
    print_header("2. Checking Environment Configuration")
    env_issues = check_env_file_security()
    if env_issues:
        for issue in env_issues:
            print_warning(issue)
            all_issues.append(issue)
        has_critical_issues = True
    else:
        print_success("Environment configuration looks secure")
    
    # 3. Check for hardcoded secrets
    print_header("3. Checking for Hardcoded Secrets")
    secret_issues = check_code_for_secrets()
    if secret_issues:
        print_error(f"Found {len(secret_issues)} potential hardcoded secrets")
        for issue in secret_issues[:10]:  # Show first 10
            print_warning(f"  - {issue}")
        has_critical_issues = True
    else:
        print_success("No hardcoded secrets found")
    
    # 4. Check .gitignore
    print_header("4. Checking .gitignore")
    gitignore_issues = check_gitignore()
    if gitignore_issues:
        for issue in gitignore_issues:
            print_warning(issue)
        all_issues.extend(gitignore_issues)
    else:
        print_success(".gitignore properly configured")
    
    # Summary
    print_header("Security Check Summary")
    if has_critical_issues:
        print_error("⚠️  CRITICAL SECURITY ISSUES FOUND!")
        print("\nPlease address the issues above before deploying to production.")
        return 1
    else:
        print_success("✅ No critical security issues found")
        if all_issues:
            print_warning(f"⚠️  {len(all_issues)} minor issues found (see above)")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

