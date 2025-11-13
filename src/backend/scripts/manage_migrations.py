#!/usr/bin/env python3
"""
Database Migration Management Script
Provides convenient commands for managing Alembic migrations
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def run_command(cmd: list, description: str) -> bool:
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {description} failed")
        print(f"Exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def check_database_url() -> bool:
    """Check if DATABASE_URL is set"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("⚠️  Warning: DATABASE_URL environment variable is not set")
        print("   Using default: postgresql://postgres:postgres@localhost:5432/drilling_db")
        print("   Set DATABASE_URL to use a different database\n")
        return False
    return True


def init_alembic() -> bool:
    """Initialize Alembic (if not already initialized)"""
    alembic_dir = Path(__file__).parent.parent / "alembic"
    if (alembic_dir / "versions").exists():
        print("✅ Alembic is already initialized")
        return True
    
    print("Initializing Alembic...")
    return run_command(
        ["alembic", "init", "alembic"],
        "Initialize Alembic"
    )


def create_migration(message: str) -> bool:
    """Create a new migration"""
    if not message:
        print("❌ Error: Migration message is required")
        print("Usage: python manage_migrations.py create 'migration message'")
        return False
    
    check_database_url()
    
    return run_command(
        ["alembic", "revision", "--autogenerate", "-m", message],
        f"Create migration: {message}"
    )


def upgrade_migration(revision: str = "head") -> bool:
    """Upgrade database to a specific revision (default: head)"""
    check_database_url()
    
    return run_command(
        ["alembic", "upgrade", revision],
        f"Upgrade database to {revision}"
    )


def downgrade_migration(revision: str = "-1") -> bool:
    """Downgrade database by one revision (or to specific revision)"""
    check_database_url()
    
    return run_command(
        ["alembic", "downgrade", revision],
        f"Downgrade database to {revision}"
    )


def show_history() -> bool:
    """Show migration history"""
    return run_command(
        ["alembic", "history"],
        "Show migration history"
    )


def show_current() -> bool:
    """Show current database revision"""
    check_database_url()
    
    return run_command(
        ["alembic", "current"],
        "Show current database revision"
    )


def stamp_revision(revision: str) -> bool:
    """Stamp database with a revision without running migrations"""
    check_database_url()
    
    return run_command(
        ["alembic", "stamp", revision],
        f"Stamp database with revision {revision}"
    )


def show_help():
    """Show help message"""
    help_text = """
╔══════════════════════════════════════════════════════════════╗
║         Database Migration Management Script                 ║
╚══════════════════════════════════════════════════════════════╝

Usage: python manage_migrations.py <command> [options]

Commands:
  init                    Initialize Alembic (if not already done)
  create <message>        Create a new migration with autogenerate
  upgrade [revision]       Upgrade database (default: head)
  downgrade [revision]    Downgrade database (default: -1)
  history                 Show migration history
  current                 Show current database revision
  stamp <revision>        Stamp database with revision (no migration)
  help                    Show this help message

Examples:
  python manage_migrations.py create "add user preferences table"
  python manage_migrations.py upgrade
  python manage_migrations.py upgrade head
  python manage_migrations.py downgrade -1
  python manage_migrations.py current
  python manage_migrations.py history

Environment Variables:
  DATABASE_URL            Database connection URL
                          Format: postgresql://user:pass@host:port/dbname
                          Default: postgresql://postgres:postgres@localhost:5432/drilling_db

Notes:
  - Make sure DATABASE_URL is set or use default
  - Always backup your database before running migrations
  - Test migrations in development before applying to production
"""
    print(help_text)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    # Change to backend directory
    backend_dir = Path(__file__).parent.parent
    os.chdir(backend_dir)
    
    success = False
    
    if command == "init":
        success = init_alembic()
    elif command == "create":
        message = sys.argv[2] if len(sys.argv) > 2 else ""
        success = create_migration(message)
    elif command == "upgrade":
        revision = sys.argv[2] if len(sys.argv) > 2 else "head"
        success = upgrade_migration(revision)
    elif command == "downgrade":
        revision = sys.argv[2] if len(sys.argv) > 2 else "-1"
        success = downgrade_migration(revision)
    elif command == "history":
        success = show_history()
    elif command == "current":
        success = show_current()
    elif command == "stamp":
        revision = sys.argv[2] if len(sys.argv) > 2 else ""
        if not revision:
            print("❌ Error: Revision is required for stamp command")
            return
        success = stamp_revision(revision)
    elif command == "help" or command == "--help" or command == "-h":
        show_help()
        success = True
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python manage_migrations.py help' for usage information")
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

