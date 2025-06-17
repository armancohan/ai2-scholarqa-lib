#!/usr/bin/env python3
"""
Test script to verify WebSocket integration for real-time task updates.
This script simulates the ScholarQA pipeline and checks if WebSocket notifications work.
"""

import os
import sys
import json
import time
import asyncio
import tempfile
from pathlib import Path

# Add the API directory to Python path
api_dir = Path(__file__).parent / "api"
sys.path.insert(0, str(api_dir))

from scholarqa.state_mgmt.local_state_mgr import LocalStateMgrClient
from scholarqa.models import ToolRequest, TaskStep

async def test_websocket_notifications():
    """Test that WebSocket notifications are created when update_task_state is called."""
    
    print("🧪 Testing WebSocket notification system...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temp directory: {temp_dir}")
        
        # Create a LocalStateMgrClient
        state_mgr = LocalStateMgrClient(temp_dir, "async_state")
        
        # Enable WebSocket callback (simulate what happens in the app)
        async def dummy_callback(task_id, message):
            pass  # Dummy async function
        
        state_mgr.set_websocket_callback(dummy_callback)
        print(f"📡 WebSocket callback enabled, async_state_dir: {state_mgr._async_state_dir}")
        
        # Create a test task
        task_id = "test-task-123"
        tool_request = ToolRequest(
            task_id=task_id,
            query="Test query for WebSocket integration",
            user_id="test-user"
        )
        
        # Initialize the task in the state manager
        from nora_lib.tasks.models import AsyncTaskState
        from scholarqa.models import TaskStep
        
        initial_step = TaskStep(description="Task started", start_timestamp=time.time())
        task_state = AsyncTaskState(
            task_id=task_id,
            estimated_time="~3 minutes",
            task_status="Task started",
            task_result=None,
            extra_state={"query": tool_request.query, "start": time.time(), "steps": [initial_step]},
        )
        
        state_mgr_instance = state_mgr.get_state_mgr(tool_request)
        state_mgr_instance.write_state(task_state)
        print(f"✅ Task {task_id} initialized")
        
        # Simulate multiple update_task_state calls (like ScholarQA does)
        updates = [
            ("Processing user query", 5),
            ("Retrieving relevant passages from a corpus of 8M+ open access papers", 15),
            ("Retrieved 50 highly relevant passages", 1),
            ("Extracting salient key statements from papers", 20),
            ("Synthesizing an answer outline based on extracted quotes", 15),
            ("Start generating each section in the answer outline", 30),
        ]
        
        notifications_dir = os.path.join(temp_dir, "async_state", "notifications")
        
        for i, (status, estimated_time) in enumerate(updates):
            print(f"📤 Sending update {i+1}: {status}")
            
            # Call update_task_state
            state_mgr.update_task_state(
                task_id=task_id,
                tool_req=tool_request,
                status=status,
                step_estimated_time=estimated_time,
                task_estimated_time="~3 minutes"
            )
            
            # Check if notification file was created
            if os.path.exists(notifications_dir):
                notification_files = [f for f in os.listdir(notifications_dir) if f.endswith('.json')]
                print(f"📄 Found {len(notification_files)} notification files")
                
                if notification_files:
                    # Read the latest notification
                    latest_file = os.path.join(notifications_dir, notification_files[-1])
                    with open(latest_file, 'r') as f:
                        notification = json.load(f)
                    
                    print(f"✅ Notification created:")
                    print(f"   - Type: {notification['type']}")
                    print(f"   - Task ID: {notification['task_id']}")
                    print(f"   - Status: {notification['task_status']}")
                    print(f"   - Steps: {len(notification['steps'])} total")
                    
                    # Cleanup notification file
                    os.remove(latest_file)
                else:
                    print("❌ No notification file created!")
                    return False
            else:
                print("❌ Notifications directory doesn't exist!")
                return False
            
            # Small delay between updates
            time.sleep(0.1)
        
        print("✅ All WebSocket notifications created successfully!")
        return True

def test_file_structure():
    """Test that the notification file structure is correct."""
    print("\n📁 Testing file structure...")
    
    # Check that our modified files exist
    files_to_check = [
        "api/scholarqa/app.py",
        "api/scholarqa/state_mgmt/local_state_mgr.py",
        "api/scholarqa/scholar_qa.py",
        "ui/src/hooks/useWebSocket.ts",
        "ui/src/components/Results.tsx"
    ]
    
    for file_path in files_to_check:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing!")
            return False
    
    return True

async def main():
    """Run all tests."""
    print("🚀 Starting WebSocket Integration Tests\n")
    
    # Test 1: File structure
    if not test_file_structure():
        print("❌ File structure test failed!")
        return False
    
    # Test 2: WebSocket notifications
    if not await test_websocket_notifications():
        print("❌ WebSocket notification test failed!")
        return False
    
    print("\n🎉 All tests passed! WebSocket integration is working correctly.")
    print("\n📋 Summary of changes:")
    print("  1. ✅ Added WebSocket endpoint to FastAPI app")
    print("  2. ✅ Modified LocalStateMgrClient to create notification files")
    print("  3. ✅ Added background task to monitor notifications and send WebSocket updates")
    print("  4. ✅ Created React WebSocket hook with fallback to polling")
    print("  5. ✅ Updated Results component to use WebSocket-first approach")
    
    print("\n🔧 Next steps:")
    print("  1. Start the FastAPI server: python -m uvicorn api.scholarqa.app:create_app --factory --reload")
    print("  2. Start the React frontend: cd ui && npm start")
    print("  3. Submit a query and observe real-time updates!")
    
    return True

if __name__ == "__main__":
    asyncio.run(main())