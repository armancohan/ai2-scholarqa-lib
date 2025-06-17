#!/usr/bin/env python3
"""
Test WebSocket server to verify real-time updates work correctly.
This simulates running a query through the FastAPI app.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add the API directory to Python path
api_dir = Path(__file__).parent / "api"
sys.path.insert(0, str(api_dir))

async def test_websocket_flow():
    """Test the complete WebSocket flow by running the FastAPI server and simulating a query."""
    
    print("🧪 Testing complete WebSocket integration flow...")
    
    # Change to API directory for proper config loading
    original_dir = os.getcwd()
    api_dir = Path(__file__).parent / "api"
    os.chdir(api_dir)
    
    try:
        from scholarqa.app import create_app, websocket_manager
        from scholarqa.models import ToolRequest
        from fastapi.testclient import TestClient
        
        print("✅ Imports successful")
        
        # Create the FastAPI app
        app = create_app()
        print("✅ FastAPI app created")
        
        # Create test client
        client = TestClient(app)
        
        # Test that the WebSocket endpoint exists
        print("📡 Testing WebSocket endpoint exists...")
        
        # Try to start a WebSocket connection
        with client.websocket_connect("/ws/test-client") as websocket:
            print("✅ WebSocket connection established")
            
            # Send a task registration message
            websocket.send_json({
                "type": "register_task",
                "task_id": "test-task-456"
            })
            
            # Receive response
            data = websocket.receive_json()
            print(f"📥 Received: {data}")
            
            if data.get("type") == "task_registered":
                print("✅ Task registration successful")
            else:
                print("❌ Task registration failed")
                return False
                
        print("✅ WebSocket basic functionality working")
        
        # Now test the REST API for creating tasks
        print("📤 Testing task creation...")
        
        response = client.post("/query_corpusqa", json={
            "query": "Test query for WebSocket integration",
            "user_id": "test-user",
            "opt_in": True
        })
        
        print(f"📥 HTTP Response status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"📥 Response: {data}")
            task_id = data.get("task_id")
            if task_id:
                print(f"✅ Task created with ID: {task_id}")
                
                # Test status polling
                print("📊 Testing status polling...")
                status_response = client.post("/query_corpusqa", json={
                    "task_id": task_id
                })
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"📥 Status: {status_data}")
                    print("✅ Status polling working")
                else:
                    print(f"❌ Status polling failed: {status_response.status_code}")
                    return False
            else:
                print("❌ No task_id in response")
                return False
        else:
            print(f"❌ Task creation failed: {response.text}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_dir)


async def main():
    """Run the WebSocket integration test."""
    print("🚀 Starting WebSocket Integration Test\n")
    
    success = await test_websocket_flow()
    
    if success:
        print("\n🎉 WebSocket integration test passed!")
        print("\n📋 Next steps to debug the web interface:")
        print("  1. Check browser developer console for WebSocket connection errors")
        print("  2. Verify the React app is connecting to the correct WebSocket URL")
        print("  3. Check if CORS is configured properly for WebSocket connections")
        print("  4. Ensure the React app is registering tasks correctly")
        print("\n🔧 To debug the real interface:")
        print("  1. Open browser dev tools → Network → WS tab")
        print("  2. Submit a query and watch for WebSocket messages")
        print("  3. Check the console for any JavaScript errors")
    else:
        print("\n❌ WebSocket integration test failed!")
        print("There are issues with the basic WebSocket functionality.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())