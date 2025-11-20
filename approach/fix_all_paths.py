"""Fix all import paths and file paths in tests and debug folders.

This script updates:
1. sys.path.append to point to parent directory
2. Import statements to use relative imports or sys.path correctly
3. Config file paths to be relative to script location
4. Figure save paths to use results/ folder with proper relative paths
"""

import os
import re

def fix_file(filepath, is_test_or_debug=True):
    """Fix imports and paths in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix 1: Update sys.path.append to add parent directory
    if is_test_or_debug:
        # Replace sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        # with sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        content = re.sub(
            r'sys\.path\.append\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)',
            "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))",
            content
        )
    
    # Fix 2: Update config path to be relative
    # Replace 'approach/config_forward.json' with proper relative path
    if is_test_or_debug:
        content = re.sub(
            r"'approach/config_forward\.json'",
            "os.path.join(os.path.dirname(__file__), '..', 'config_forward.json')",
            content
        )
        content = re.sub(
            r'"approach/config_forward\.json"',
            'os.path.join(os.path.dirname(__file__), \'..\', \'config_forward.json\')',
            content
        )
    
    # Fix 3: Update figure save paths
    # Pattern 1: plt.savefig('approach/xxx.png', ...)
    def fix_savefig(match):
        filename = match.group(1)
        return f"plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'results', '{filename}'), "
    
    content = re.sub(
        r"plt\.savefig\('approach/([^']+\.png)',\s*",
        fix_savefig,
        content
    )
    
    # Pattern 2: print statements mentioning the path
    def fix_print_path(match):
        filename = match.group(1)
        return f'f"  Plot saved: {{os.path.join(os.path.dirname(__file__), \'..\', \'results\', \'{filename}\')}}"'
    
    content = re.sub(
        r"'approach/([^']+\.png)'(?=\))",
        lambda m: f"os.path.join(os.path.dirname(__file__), '..', 'results', '{m.group(1)}')",
        content
    )
    
    # Only write if changes were made
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix all Python files in tests and debug folders."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    tests_dir = os.path.join(script_dir, 'tests')
    debug_dir = os.path.join(script_dir, 'debug')
    
    fixed_files = []
    
    # Fix test files
    if os.path.exists(tests_dir):
        for filename in os.listdir(tests_dir):
            if filename.endswith('.py'):
                filepath = os.path.join(tests_dir, filename)
                if fix_file(filepath, is_test_or_debug=True):
                    fixed_files.append(f"tests/{filename}")
    
    # Fix debug files
    if os.path.exists(debug_dir):
        for filename in os.listdir(debug_dir):
            if filename.endswith('.py'):
                filepath = os.path.join(debug_dir, filename)
                if fix_file(filepath, is_test_or_debug=True):
                    fixed_files.append(f"debug/{filename}")
    
    if fixed_files:
        print(f"Fixed {len(fixed_files)} files:")
        for f in fixed_files:
            print(f"  ✓ {f}")
    else:
        print("No files needed fixing.")
    
    print("\nDone! All import and path issues should be resolved.")

if __name__ == "__main__":
    main()
