"""
Quick installation verification script for LLM integration.
Run this after installing dependencies to verify everything works.
"""

import sys
import importlib

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False

def main():
    print("="*60)
    print("LLM Integration - Installation Verification")
    print("="*60)
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    deps = [
        "pytz",
        "psutil",
        "ollama",
        "aiohttp"
    ]
    
    all_ok = True
    for dep in deps:
        if not check_import(dep):
            all_ok = False
    
    print()
    
    # Check LLM modules
    print("Checking LLM modules...")
    modules = [
        "app.llm.gpu_scheduler",
        "app.llm.system_config",
        "app.llm.monitor"
    ]
    
    for mod in modules:
        if not check_import(mod):
            all_ok = False
    
    print()
    
    # Try to initialize
    print("Testing initialization...")
    try:
        from app.llm.gpu_scheduler import gpu_scheduler
        from app.llm.system_config import system_config
        
        print(f"✅ GPU Scheduler initialized")
        print(f"✅ System Config initialized")
        
        # Get config summary
        config = system_config.get_config_summary()
        print(f"\nCurrent configuration:")
        print(f"  Market session: {config['market_session']}")
        print(f"  GPU mode: {config['gpu_mode']}")
        print(f"  CPU load: {config['cpu_load']}%")
        print(f"  RAM usage: {config['memory_usage']['percent']:.1f}%")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        all_ok = False
    
    print()
    print("="*60)
    if all_ok:
        print("✅ ALL CHECKS PASSED")
        print()
        print("Next steps:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Pull model: ollama pull qwen3-30b-a3b:q4_K_M")
        print("3. Run tests: python -m app.llm.test_scheduling both")
    else:
        print("❌ SOME CHECKS FAILED")
        print()
        print("Fix issues and run this script again.")
        sys.exit(1)
    print("="*60)

if __name__ == "__main__":
    main()
