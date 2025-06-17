#!/usr/bin/env python3
"""
Test a real query to see if notification files are created during actual processing.
"""

import asyncio
import os
import time
import json
import glob
from pathlib import Path

async def test_real_query_notifications():
    """Test that a real query creates notification files."""
    
    print("🧪 Testing real query notification creation...")
    
    # Change to API directory
    api_dir = Path(__file__).parent / "api"
    os.chdir(api_dir)
    
    try:
        from fastapi.testclient import TestClient
        from scholarqa.app import create_app
        
        print("✅ Imports successful")
        
        # Create the FastAPI app and test client
        app = create_app()
        client = TestClient(app)
        
        print("✅ FastAPI app created")
        
        # Clear any existing notification files
        notifications_dir = Path("logs/async_state/notifications")
        if notifications_dir.exists():
            for file in notifications_dir.glob("*.json"):
                file.unlink()
            print("🧹 Cleared existing notification files")
        
        # Submit a simple query
        print("📤 Submitting query...")
        response = client.post("/query_corpusqa", json={
            "query": "WebSocket",  # Simple one-word query to make it faster
            "user_id": "test-user",
            "opt_in": True
        })
        
        if response.status_code != 200:
            print(f"❌ Query submission failed: {response.status_code}")
            print(response.text)
            return False
        
        data = response.json()
        task_id = data.get("task_id")
        print(f"✅ Query submitted, task ID: {task_id}")
        
        # Monitor for notification files for a short time
        print("👀 Monitoring for notification files...")
        
        start_time = time.time()
        notification_count = 0
        
        while time.time() - start_time < 30:  # Monitor for 30 seconds
            if notifications_dir.exists():
                current_files = list(notifications_dir.glob("*.json"))
                
                if len(current_files) > notification_count:
                    new_files = current_files[notification_count:]
                    notification_count = len(current_files)
                    
                    for file_path in new_files:
                        try:
                            with open(file_path, 'r') as f:
                                notification = json.load(f)
                            print(f"📄 New notification: {notification.get('task_status', 'Unknown')}")
                        except Exception as e:
                            print(f"❌ Error reading {file_path}: {e}")
            
            await asyncio.sleep(0.5)
        
        if notification_count > 0:
            print(f"✅ Found {notification_count} notification files during query processing!")
            return True
        else:
            print("❌ No notification files were created during query processing")
            
            # Check if task is still running
            status_response = client.post("/query_corpusqa", json={"task_id": task_id})
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"📊 Task status: {status_data.get('task_status', 'Unknown')}")
            
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("🚀 Real Query Notification Test\n")
    
    success = await test_real_query_notifications()
    
    if success:
        print("\n🎉 Real query creates notifications!")
        print("The issue is likely in the WebSocket broadcasting or frontend connection.")
    else:
        print("\n❌ Real query does not create notifications!")
        print("The issue is in the multiprocess setup or state manager configuration.")

if __name__ == "__main__":
    asyncio.run(main())