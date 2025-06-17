import { useEffect, useRef, useState } from 'react';
import { AsyncTaskState } from '../@types/AsyncTaskState';
import { formatStatus } from '../api/utils';

export interface WebSocketMessage {
  type: 'task_update' | 'task_registered' | 'error';
  task_id?: string;
  task_status?: string;
  estimated_time?: string;
  current_step?: any;
  steps?: any[];
  partial_result?: any;
  message?: string;
}

export interface UseWebSocketReturn {
  isConnected: boolean;
  lastMessage: WebSocketMessage | null;
  taskState: AsyncTaskState | null;
  registerTask: (taskId: string) => void;
  disconnect: () => void;
}

export const useWebSocket = (clientId: string): UseWebSocketReturn => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [taskState, setTaskState] = useState<AsyncTaskState | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const maxReconnectAttempts = 5;
  const reconnectAttempts = useRef(0);

  const connect = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/ws/${clientId}`;
    
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      reconnectAttempts.current = 0;
    };

    wsRef.current.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        setLastMessage(message);
        
        if (message.type === 'task_update' && message.task_id) {
          // Convert WebSocket message to AsyncTaskState format
          const asyncTaskState: AsyncTaskState = {
            task_id: message.task_id,
            query: taskState?.query || '',
            task_status: message.task_status || '',
            estimated_time: message.estimated_time || '',
            task_result: message.partial_result || taskState?.task_result || null,
            steps: message.steps || []
          };
          
          setTaskState(formatStatus(asyncTaskState));
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    wsRef.current.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      
      // Attempt to reconnect if we haven't exceeded max attempts
      if (reconnectAttempts.current < maxReconnectAttempts) {
        reconnectAttempts.current++;
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
        console.log(`Attempting to reconnect in ${delay}ms (attempt ${reconnectAttempts.current})`);
        
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, delay);
      }
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  };

  const disconnect = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setIsConnected(false);
    reconnectAttempts.current = maxReconnectAttempts; // Prevent reconnection
  };

  const registerTask = (taskId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'register_task',
        task_id: taskId
      }));
    }
  };

  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, [clientId]);

  return {
    isConnected,
    lastMessage,
    taskState,
    registerTask,
    disconnect
  };
};