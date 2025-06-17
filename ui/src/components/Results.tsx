import React, { useEffect, useState } from 'react';
import {  Typography } from '@mui/material';
import { updateStatus } from '../api/utils';
import { Sections } from './Sections';
import { historyType } from './shared';
import { AsyncTaskState, TaskStatus, TaskStep } from '../@types/AsyncTaskState';
import { StepProgress, StepProgressPropType } from './StepProgress';
import { useWebSocket } from '../hooks/useWebSocket';

const DEFAULT_INTERVAL = 3000;

interface PropType {
  taskId: string;
  interval?: number;
  history: historyType;
  setHistory: (history: historyType) => void;
  cookieUserId: string
}

function isTaskRunning(status: AsyncTaskState | undefined): boolean {
  return Boolean(status?.task_status && typeof status?.task_status === 'string' && status.estimated_time && typeof status.estimated_time === 'string');
}

export const Results: React.FC<PropType> = (props) => {
  const { taskId, interval = DEFAULT_INTERVAL, history, setHistory, cookieUserId } = props;

  const [status, setStatus] = useState<AsyncTaskState | undefined>();
  const [httpStatus, setHttpStatus] = useState<number>(200);
  const [usePolling, setUsePolling] = useState<boolean>(false);

  const [progressProps, setProgressProps] = useState<StepProgressPropType>({
    estimatedTime: 'Loading...',
    steps: [],
  })

  // Initialize WebSocket connection
  const { isConnected, taskState, registerTask, disconnect } = useWebSocket(cookieUserId || 'anonymous');

  // Register task with WebSocket when connected
  useEffect(() => {
    if (isConnected && taskId && !usePolling) {
      registerTask(taskId);
    }
  }, [isConnected, taskId, registerTask, usePolling]);

  // Handle WebSocket task state updates
  useEffect(() => {
    if (taskState && !usePolling) {
      setStatus(taskState);
      setHttpStatus(200);
      
      const task_status = taskState?.task_status as undefined | TaskStatus;
      const estimated_time = taskState?.estimated_time;
      const taskRunning = isTaskRunning(taskState);
      
      if (taskRunning && typeof task_status === 'string' && typeof estimated_time === 'string' && taskState.steps) {
        setProgressProps({
          estimatedTime: estimated_time ?? 'Loading...',
          steps: taskState.steps as TaskStep[]
        })
      } else if (!taskRunning) {
        setProgressProps({
          estimatedTime: 'Complete',
          steps: taskState.steps as TaskStep[] || []
        })
      }
    }
  }, [taskState, usePolling]);

  // Fallback to polling if WebSocket fails
  useEffect(() => {
    if (!isConnected && !usePolling) {
      console.log('WebSocket not connected, falling back to polling');
      setUsePolling(true);
    }
  }, [isConnected, usePolling]);

  // Polling logic (fallback)
  useEffect(() => {
    if (!usePolling) return;
    
    const timeoutIds: number[] = [];

    const inner = async () => {
      console.log('polling - results', taskId);
      const { update, httpStatus } = await updateStatus(taskId);
      setHttpStatus(httpStatus);
      setStatus(update);
      const taskRunning = isTaskRunning(update);

      if (httpStatus !== 200) {
        setProgressProps({
          estimatedTime: 'Error',
          error: `Something went wrong - please try a different query. ${update?.task_status ?? update?.detail ?? 'Unknown error'}`,
          steps: []
        })
      } else {
        try {
          const task_status = update?.task_status as undefined | TaskStatus;
          const estimated_time = update?.estimated_time;
          if (taskRunning && typeof task_status === 'string' && typeof estimated_time === 'string' && update.steps) {
            setProgressProps({
              estimatedTime: estimated_time ?? 'Loading...',
              steps: update.steps as TaskStep[]
            })
          } else {
            setProgressProps({
              estimatedTime: taskRunning ? 'Loading...' : 'Complete',
              steps: update.steps as TaskStep[] || []
            })
          }
        } catch (e) {
          console.error('error parsing status', e);
        }
        if (taskRunning) {
          const timeoutId = window.setTimeout(inner, interval);
          timeoutIds.push(timeoutId);
        }
      }
    }
    inner();
    return () => {
      timeoutIds.forEach(clearTimeout);
    }
  }, [taskId, interval, usePolling]);

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  const taskRunning = isTaskRunning(status);
  useEffect(() => {
    if (!taskRunning && !history[taskId] && status?.query && httpStatus === 200) {
      setHistory({
        ...(history ?? {}),
        [taskId]: {
          query: status.query, taskId: taskId, timestamp: Date.now()
        }
      });
    } else {
      console.log('not adding back', taskRunning, history[taskId], status?.query, httpStatus)
    }
  }, [taskRunning, history, taskId, status?.query, httpStatus, setHistory])
  const sections = status?.task_result?.sections ?? [];

  return (
    <>
      <Typography variant="h3" sx={{ marginBottom: '16px' }}>{status?.query ?? ''}</Typography>
      {sections.length > 0 && (
        <Sections sections={sections} taskId={taskId} cookieUserId={cookieUserId} />
      )}
      {(taskRunning || httpStatus !== 200) && <StepProgress {...progressProps} />}
    </>
  );
};
