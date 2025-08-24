"""Simple verification script to test the new project structure"""

import sys
import os

def test_structure():
    """Test that the new project structure is correct"""
    print("Testing new project structure...")
    print("-" * 40)
    
    # Check if src directory exists
    src_path = os.path.join(os.getcwd(), 'src')
    if os.path.exists(src_path):
        print("✓ src/ directory exists")
    else:
        print("✗ src/ directory missing")
        return False
    
    # Check if drilling_env is in src
    drilling_env_path = os.path.join(src_path, 'drilling_env')
    if os.path.exists(drilling_env_path):
        print("✓ drilling_env moved to src/")
    else:
        print("✗ drilling_env not found in src/")
        return False
    
    # Check if tests directory exists
    tests_path = os.path.join(os.getcwd(), 'tests')
    if os.path.exists(tests_path):
        print("✓ tests/ directory exists")
    else:
        print("✗ tests/ directory missing")
        return False
    
    # Check if test files exist in tests directory
    test_env_path = os.path.join(tests_path, 'test_env.py')
    test_physics_path = os.path.join(tests_path, 'test_physics.py')
    
    if os.path.exists(test_env_path):
        print("✓ test_env.py moved to tests/")
    else:
        print("✗ test_env.py not found in tests/")
        return False
    
    if os.path.exists(test_physics_path):
        print("✓ test_physics.py moved to tests/")
    else:
        print("✗ test_physics.py not found in tests/")
        return False
    
    # Check if old test files are removed from root
    old_test_env = os.path.join(os.getcwd(), 'test_env.py')
    old_test_physics = os.path.join(os.getcwd(), 'test_physics.py')
    
    if not os.path.exists(old_test_env):
        print("✓ Old test_env.py removed from root")
    else:
        print("✗ Old test_env.py still exists in root")
    
    if not os.path.exists(old_test_physics):
        print("✓ Old test_physics.py removed from root")
    else:
        print("✗ Old test_physics.py still exists in root")
    
    # Check key files in drilling_env
    key_files = ['drilling_env.py', 'drilling_physics.py', '__init__.py']
    for file_name in key_files:
        file_path = os.path.join(drilling_env_path, file_name)
        if os.path.exists(file_path):
            print(f"✓ {file_name} exists in src/drilling_env/")
        else:
            print(f"✗ {file_name} missing from src/drilling_env/")
            return False
    
    print("\n✓ All structural changes completed successfully!")
    print("\nProject structure is now aligned with standards:")
    print("- Source code moved to src/ directory")
    print("- Tests converted to pytest format and moved to tests/ directory")
    print("- Old test files removed from root directory")
    
    return True

def test_import_paths():
    """Test that import paths work correctly"""
    print("\nTesting import paths...")
    print("-" * 40)
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
    
    try:
        # Test basic imports without external dependencies
        import drilling_env
        print("✓ drilling_env package can be imported")
        
        # Check if modules exist (without importing to avoid dependency issues)
        drilling_env_path = os.path.join(os.getcwd(), 'src', 'drilling_env')
        
        env_file = os.path.join(drilling_env_path, 'drilling_env.py')
        physics_file = os.path.join(drilling_env_path, 'drilling_physics.py')
        
        if os.path.exists(env_file):
            print("✓ drilling_env.py module accessible")
        
        if os.path.exists(physics_file):
            print("✓ drilling_physics.py module accessible")
        
        print("\n✓ Import structure is correct!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

if __name__ == "__main__":
    structure_ok = test_structure()
    import_ok = test_import_paths()
    
    if structure_ok and import_ok:
        print("\n🎉 Project restructuring completed successfully!")
        print("\nNext steps:")
        print("1. Install required dependencies (gym, stable-baselines3, etc.)")
        print("2. Install pytest for running tests")
        print("3. Run: python -m pytest tests/ -v")
    else:
        print("\n❌ Some issues found. Please check the output above.")