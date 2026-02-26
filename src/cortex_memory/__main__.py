"""Allow `python3 -m cortex_memory install` as an alternative entry point."""

import sys

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "install":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from cortex_memory.install import main as install_main
        install_main()
    else:
        print("Usage:")
        print("  python3 -m cortex_memory install [--global] [--project]")
        print()
        print("Or use the library directly:")
        print("  from cortex_memory import Memory")
        print("  mem = Memory.create('my project')")

if __name__ == "__main__":
    main()
