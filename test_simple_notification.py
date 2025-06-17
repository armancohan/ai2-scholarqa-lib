#!/usr/bin/env python3
"""
Simple test to check if notification files are created when update_task_state is called.
"""

import os
import sys
import tempfile
import time
import json
from pathlib import Path

# Add the API directory to Python path
api_dir = Path(__file__).parent / "api"
sys.path.insert(0, str(api_dir))

def test_notifications():
    """Test that notification files are created."""
    
    print("🧪 Testing notification file creation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temp directory: {temp_dir}")
        
        from scholarqa.state_mgmt.local_state_mgr import LocalStateMgrClient
        from scholarqa.models import ToolRequest, TaskStep
        from nora_lib.tasks.models import AsyncTaskState
        
        # Create a LocalStateMgrClient
        state_mgr = LocalStateMgrClient(temp_dir, "async_state")
        print(f"📁 State dir: {state_mgr._async_state_dir}")
        
        # Create a test task
        task_id = "test-notification-task"
        tool_request = ToolRequest(
            task_id=task_id,
            query="Test query for notification",
            user_id="test-user"
        )
        
        # Initialize task state
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
        print("✅ Task initialized")
        
        # Call update_task_state
        print("📤 Calling update_task_state...")
        state_mgr.update_task_state(
            task_id=task_id,
            tool_req=tool_request,
            status="Testing notification creation",
            step_estimated_time=10,
            task_estimated_time="~2 minutes"
        )
        
        # Check for notification files
        notifications_dir = os.path.join(temp_dir, "async_state", "notifications")
        print(f"🔍 Checking {notifications_dir}")
        
        if os.path.exists(notifications_dir):
            notification_files = [f for f in os.listdir(notifications_dir) if f.endswith('.json')]
            print(f"📄 Found {len(notification_files)} notification files")
            
            if notification_files:
                # Read the notification
                latest_file = os.path.join(notifications_dir, notification_files[0])
                with open(latest_file, 'r') as f:
                    notification = json.load(f)
                
                print("✅ Notification created successfully!")
                print(f"   📋 Type: {notification['type']}")
                print(f"   🆔 Task ID: {notification['task_id']}")
                print(f"   📊 Status: {notification['task_status']}")
                print(f"   📈 Steps: {len(notification['steps'])} total")
                return True
            else:
                print("❌ No notification files found!")
                return False
        else:
            print("❌ Notifications directory doesn't exist!")
            return False


if __name__ == "__main__":
    print("🚀 Simple Notification Test\n")
    
    success = test_notifications()
    
    if success:
        print("\n🎉 Notification system is working!")
        print("\n📋 This means:")
        print("  ✅ Notification files are created when update_task_state is called")
        print("  ✅ The state manager is properly configured")
        print("  ✅ The WebSocket monitor should pick up these files")
        print("\n🔍 Next: Check if WebSocket monitor is running and broadcasting")
    else:
        print("\n❌ Notification system is not working!")
        print("The issue is in the state manager or file creation logic.")