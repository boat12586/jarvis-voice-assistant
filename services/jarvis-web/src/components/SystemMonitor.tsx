'use client';

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { HealthStatus, SystemStats } from '@/types/jarvis';
import HolographicPanel from './HolographicPanel';

interface SystemMonitorProps {
  healthStatus?: HealthStatus;
  systemStats?: SystemStats;
}

const SystemMonitor: React.FC<SystemMonitorProps> = ({ healthStatus, systemStats }) => {
  const [cpuUsage, setCpuUsage] = useState(0);
  const [memoryUsage, setMemoryUsage] = useState(0);
  const [networkActivity, setNetworkActivity] = useState(0);

  // Simulate real-time metrics
  useEffect(() => {
    const interval = setInterval(() => {
      setCpuUsage(prev => Math.max(0, Math.min(100, prev + (Math.random() - 0.5) * 10)));
      setMemoryUsage(prev => Math.max(0, Math.min(100, prev + (Math.random() - 0.5) * 5)));
      setNetworkActivity(Math.random() * 100);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-400';
      case 'degraded': return 'text-yellow-400';
      case 'unhealthy': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const MetricBar = ({ label, value, color, unit = '%' }: { 
    label: string; 
    value: number; 
    color: string; 
    unit?: string;
  }) => (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-gray-300">{label}</span>
        <span className={color}>{value.toFixed(1)}{unit}</span>
      </div>
      <div className="h-2 bg-gray-700/50 rounded-full overflow-hidden">
        <motion.div
          className={`h-full ${color.replace('text-', 'bg-')} rounded-full`}
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 0.5 }}
        />
      </div>
    </div>
  );

  return (
    <div className="space-y-4">
      {/* System Health Overview */}
      <HolographicPanel title="System Status" glowColor="green" variant="glass">
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-gray-300 text-sm">Overall Status</span>
            <span className={`font-mono text-sm uppercase ${getStatusColor(healthStatus?.status || 'unknown')}`}>
              {healthStatus?.status || 'Unknown'}
            </span>
          </div>
          
          {healthStatus && (
            <div className="grid grid-cols-2 gap-3 text-xs">
              {Object.entries(healthStatus.services).map(([service, status]) => (
                <div key={service} className="flex items-center justify-between">
                  <span className="text-gray-400 capitalize">{service}</span>
                  <div className="flex items-center space-x-1">
                    <div className={`w-2 h-2 rounded-full ${
                      status === 'healthy' ? 'bg-green-400' :
                      status === 'degraded' ? 'bg-yellow-400' :
                      status === 'unhealthy' ? 'bg-red-400' : 'bg-gray-400'
                    }`} />
                    <span className={getStatusColor(status)}>{status}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </HolographicPanel>

      {/* Real-time Metrics */}
      <HolographicPanel title="Performance Metrics" glowColor="cyan" variant="glass">
        <div className="space-y-4">
          <MetricBar label="CPU Usage" value={cpuUsage} color="text-cyan-400" />
          <MetricBar label="Memory Usage" value={memoryUsage} color="text-blue-400" />
          <MetricBar label="Network Activity" value={networkActivity} color="text-purple-400" />
        </div>
      </HolographicPanel>

      {/* Connection Stats */}
      {systemStats && (
        <HolographicPanel title="Active Sessions" glowColor="yellow" variant="glass">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-gray-300 text-sm">Active Connections</span>
              <span className="text-yellow-400 font-mono">{systemStats.active_connections}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-300 text-sm">Total Sessions</span>
              <span className="text-yellow-400 font-mono">{systemStats.total_sessions}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-300 text-sm">Redis Status</span>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  systemStats.redis_connected ? 'bg-green-400' : 'bg-red-400'
                }`} />
                <span className={systemStats.redis_connected ? 'text-green-400' : 'text-red-400'}>
                  {systemStats.redis_connected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
            {systemStats.uptime && (
              <div className="flex items-center justify-between">
                <span className="text-gray-300 text-sm">Uptime</span>
                <span className="text-gray-400 font-mono text-xs">{systemStats.uptime}</span>
              </div>
            )}
          </div>
        </HolographicPanel>
      )}

      {/* Network Graph */}
      <HolographicPanel title="Network Activity" glowColor="purple" variant="glass">
        <div className="h-20 flex items-end justify-center space-x-1">
          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              className="w-1 bg-purple-400 rounded-t-sm"
              initial={{ height: 0 }}
              animate={{ 
                height: `${Math.random() * 100}%`,
                opacity: 0.3 + Math.random() * 0.7
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                repeatType: 'reverse',
                delay: i * 0.1
              }}
            />
          ))}
        </div>
      </HolographicPanel>
    </div>
  );
};

export default SystemMonitor;