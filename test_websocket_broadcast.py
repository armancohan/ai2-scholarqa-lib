#!/usr/bin/env python3
"""
Test WebSocket broadcasting by manually creating notification files and checking if they're broadcast.
"""

import asyncio
import json
import os
import time
import websockets
from pathlib import Path

async def test_websocket_broadcasting():
    """Test that notification files are picked up and broadcast via WebSocket."""
    
    print("🧪 Testing WebSocket broadcasting...")
    
    # Start the FastAPI server in the background (we'll assume it's running)
    print("📢 Make sure the FastAPI server is running:")
    print("   cd api && python -m uvicorn scholarqa.app:create_app --factory --reload")
    print("\nPress Enter when the server is running...")
    input()
    
    try:
        # Connect to WebSocket
        websocket_url = "ws://localhost:8000/api/ws/test-broadcast-client"
        print(f"🔌 Connecting to {websocket_url}")
        
        async with websockets.connect(websocket_url) as websocket:
            print("✅ WebSocket connected")
            
            # Register for a test task
            test_task_id = "broadcast-test-task"
            await websocket.send(json.dumps({
                "type": "register_task",
                "task_id": test_task_id
            }))
            
            # Wait for registration confirmation
            response = await websocket.recv()
            data = json.loads(response)
            print(f"📥 Registration response: {data}")
            
            if data.get("type") != "task_registered":
                print("❌ Task registration failed")
                return False
            
            print("✅ Task registered for WebSocket updates")
            
            # Create a notification file manually
            notifications_dir = Path("api/logs/async_state/notifications")
            notifications_dir.mkdir(parents=True, exist_ok=True)
            
            notification_data = {
                "type": "task_update",
                "task_id": test_task_id,
                "task_status": "Manual test notification",
                "estimated_time": "~1 minute",
                "current_step": {
                    "description": "Manual test notification",
                    "start_timestamp": time.time(),
                    "estimated_timestamp": time.time() + 60
                },
                "steps": [
                    {
                        "description": "Test step 1",
                        "start_timestamp": time.time() - 10,
                        "estimated_timestamp": time.time() - 5
                    },
                    {
                        "description": "Manual test notification",
                        "start_timestamp": time.time(),
                        "estimated_timestamp": time.time() + 60
                    }
                ]
            }
            
            notification_file = notifications_dir / f"{test_task_id}_{time.time()}.json"
            with open(notification_file, 'w') as f:
                json.dump(notification_data, f)
            
            print(f"📄 Created notification file: {notification_file.name}")
            
            # Wait for WebSocket message
            print("👂 Waiting for WebSocket broadcast...")
            
            try:
                # Wait up to 5 seconds for a message
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                received_data = json.loads(message)
                
                print("✅ Received WebSocket message!")
                print(f"   📋 Type: {received_data.get('type')}")
                print(f"   🆔 Task ID: {received_data.get('task_id')}")
                print(f"   📊 Status: {received_data.get('task_status')}")
                
                if (received_data.get('task_id') == test_task_id and 
                    received_data.get('task_status') == "Manual test notification"):
                    print("🎉 WebSocket broadcasting is working correctly!")
                    return True
                else:
                    print("❌ Received message doesn't match expected content")
                    return False
                    
            except asyncio.TimeoutError:
                print("❌ No WebSocket message received within 5 seconds")
                print("💡 The notification monitor might not be running or working")
                return False
                
    except Exception as e:
        print(f"❌ Error during WebSocket test: {e}")
        return False

async def main():
    print("🚀 WebSocket Broadcasting Test\n")
    
    success = await test_websocket_broadcasting()
    
    if success:
        print("\n🎉 WebSocket broadcasting is working!")
        print("The issue must be in the React frontend connection or registration.")
    else:
        print("\n❌ WebSocket broadcasting is not working!")
        print("Check if:")
        print("  - FastAPI server is running with the lifespan monitor")
        print("  - WebSocket endpoint is accessible")
        print("  - Notification monitor background task is running")

if __name__ == "__main__":
    asyncio.run(main())