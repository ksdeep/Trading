"""
DEBUG SCRIPT - Shows pickle file structure
Run: python debug_pickle.py
"""

import pickle
from pathlib import Path
from GetFreshMarketData import *

def debug_pickle(file_path):
    """Debug script to check pickle file structure"""
    print("=" * 70)
    print("🔍 MONTE CARLO PICKLE FILE DEBUG")
    print("=" * 70)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n✅ Successfully loaded: {file_path}")
        print(f"\n📊 Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"\n🔑 Available keys/items:")
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"   - {key}: list with {len(value)} items")
                    if len(value) > 0:
                        print(f"     First item type: {type(value[0])}")
                        if isinstance(value[0], (int, float)):
                            print(f"     First item value: {value[0]}")
                        elif isinstance(value[0], (list, tuple)):
                            print(f"     First item value: {value[0][:5]}... (truncated)")
                else:
                    print(f"   - {key}: {type(value)}")
        
        print("\n✅ Debug complete!")
        
    except FileNotFoundError:
        print(f"\n❌ File not found: {file_path}")
        print("Make sure the file path is correct.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    # Change this to your file path
    MC_FILE = TEMP/"monte_carlo_results.pkl"
    # OR use absolute path:
    # MC_FILE = Path(r"c:\Users\ksdee\Documents\Trading\Data\temp\monte_carlo_results.pkl")
    
    debug_pickle(MC_FILE)
