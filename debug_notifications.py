#!/usr/bin/env python3
"""
Debug script to check if notification files are being created during query processing.
"""

import os
import time
import subprocess
import sys
from pathlib import Path

def monitor_notifications():
    """Monitor the notifications directory for file creation."""
    notifications_dir = Path("api/logs/async_state/notifications")
    
    print(f"🔍 Monitoring {notifications_dir}")
    print("📁 Creating directory if it doesn't exist...")
    notifications_dir.mkdir(parents=True, exist_ok=True)
    
    print("👀 Watching for notification files...")
    print("Press Ctrl+C to stop monitoring\n")
    
    seen_files = set()
    
    try:
        while True:
            if notifications_dir.exists():
                current_files = set(notifications_dir.glob("*.json"))
                
                # Check for new files
                new_files = current_files - seen_files
                if new_files:
                    for file_path in new_files:
                        print(f"📄 New notification file: {file_path.name}")
                        try:
                            import json
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            print(f"   📋 Content: {data.get('task_status', 'Unknown status')}")
                            print(f"   🆔 Task ID: {data.get('task_id', 'Unknown')}")
                        except Exception as e:
                            print(f"   ❌ Error reading file: {e}")
                
                seen_files = current_files
                
                # Clean up old files to prevent clutter
                for file_path in current_files:
                    if file_path.stat().st_mtime < time.time() - 60:  # Older than 1 minute
                        print(f"🧹 Cleaning up old file: {file_path.name}")
                        file_path.unlink()
                        
            time.sleep(0.5)  # Check every 500ms
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")


def check_state_manager_setup():
    """Check if the state manager is properly configured."""
    print("🔧 Checking state manager configuration...")
    
    try:
        # Change to API directory
        api_dir = Path("api")
        os.chdir(api_dir)
        
        sys.path.insert(0, str(Path.cwd()))
        
        from scholarqa.app import app_config
        from scholarqa.state_mgmt.local_state_mgr import LocalStateMgrClient
        
        print("✅ Imports successful")
        
        # Check if state manager client exists
        if hasattr(app_config, 'state_mgr_client') and app_config.state_mgr_client:
            print("✅ State manager client exists")
            print(f"📁 Async state dir: {app_config.state_mgr_client._async_state_dir}")
            print(f"🔌 WebSocket callback set: {app_config.state_mgr_client.websocket_callback is not None}")
        else:
            print("❌ State manager client not found")
            
        # Test creating a state manager manually
        test_state_mgr = LocalStateMgrClient("logs", "async_state")
        print(f"📁 Test state manager dir: {test_state_mgr._async_state_dir}")
        
        # Test setting callback
        async def dummy_callback(task_id, message):
            pass
        
        test_state_mgr.set_websocket_callback(dummy_callback)
        print(f"🔌 Test callback set: {test_state_mgr.websocket_callback is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking state manager: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir("..")


if __name__ == "__main__":
    print("🚀 Debug Notification System\n")
    
    # First check the state manager setup
    if not check_state_manager_setup():
        print("\n❌ State manager setup failed. Cannot proceed.")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("\n📋 Instructions:")
    print("1. Keep this script running")
    print("2. In another terminal, start the FastAPI server:")
    print("   cd api && python -m uvicorn scholarqa.app:create_app --factory --reload")
    print("3. In a third terminal, submit a query via curl or the web interface")
    print("4. Watch this script for notification file creation")
    print("\n💡 If no files appear, the issue is in the state manager setup")
    print("💡 If files appear but WebSocket doesn't work, the issue is in the WebSocket broadcasting")
    print("\n" + "="*50 + "\n")
    
    monitor_notifications()