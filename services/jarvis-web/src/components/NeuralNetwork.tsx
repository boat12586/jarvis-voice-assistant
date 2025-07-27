'use client';

import React, { useEffect, useRef, useState } from 'react';

interface NeuralNetworkProps {
  className?: string;
  isActive?: boolean;
  activityLevel?: number; // 0-1
  networkSize?: 'small' | 'medium' | 'large';
  theme?: 'cyan' | 'purple' | 'green' | 'orange';
}

interface Node {
  id: string;
  x: number;
  y: number;
  layer: number;
  activation: number;
  baseActivation: number;
  connections: Connection[];
}

interface Connection {
  from: string;
  to: string;
  weight: number;
  activity: number;
}

const NeuralNetwork: React.FC<NeuralNetworkProps> = ({
  className = '',
  isActive = false,
  activityLevel = 0.3,
  networkSize = 'medium',
  theme = 'cyan'
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const [nodes, setNodes] = useState<Node[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);

  // Network configuration based on size
  const networkConfig = {
    small: { layers: [3, 4, 3], width: 300, height: 200 },
    medium: { layers: [4, 6, 4, 2], width: 400, height: 250 },
    large: { layers: [6, 8, 6, 4, 2], width: 500, height: 300 }
  };

  const config = networkConfig[networkSize];

  // Theme colors
  const themes = {
    cyan: {
      node: '#06b6d4',
      connection: '#67e8f9',
      active: '#22d3ee',
      pulse: '#0891b2'
    },
    purple: {
      node: '#8b5cf6',
      connection: '#c4b5fd',
      active: '#a78bfa',
      pulse: '#7c3aed'
    },
    green: {
      node: '#10b981',
      connection: '#6ee7b7',
      active: '#34d399',
      pulse: '#059669'
    },
    orange: {
      node: '#f59e0b',
      connection: '#fbbf24',
      active: '#fcd34d',
      pulse: '#d97706'
    }
  };

  const colors = themes[theme];

  // Initialize network structure
  useEffect(() => {
    const newNodes: Node[] = [];
    const newConnections: Connection[] = [];

    // Create nodes
    config.layers.forEach((layerSize, layerIndex) => {
      const layerX = (layerIndex / (config.layers.length - 1)) * (config.width - 60) + 30;
      
      for (let nodeIndex = 0; nodeIndex < layerSize; nodeIndex++) {
        const nodeY = ((nodeIndex + 1) / (layerSize + 1)) * config.height;
        const nodeId = `${layerIndex}-${nodeIndex}`;
        
        newNodes.push({
          id: nodeId,
          x: layerX,
          y: nodeY,
          layer: layerIndex,
          activation: Math.random() * 0.3,
          baseActivation: Math.random() * 0.3,
          connections: []
        });
      }
    });

    // Create connections
    for (let layerIndex = 0; layerIndex < config.layers.length - 1; layerIndex++) {
      const currentLayer = newNodes.filter(n => n.layer === layerIndex);
      const nextLayer = newNodes.filter(n => n.layer === layerIndex + 1);

      currentLayer.forEach(fromNode => {
        nextLayer.forEach(toNode => {
          const connectionId = `${fromNode.id}-${toNode.id}`;
          const connection: Connection = {
            from: fromNode.id,
            to: toNode.id,
            weight: (Math.random() - 0.5) * 2, // -1 to 1
            activity: 0
          };
          
          newConnections.push(connection);
          fromNode.connections.push(connection);
        });
      });
    }

    setNodes(newNodes);
    setConnections(newConnections);
  }, [networkSize]);

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const setCanvasSize = () => {
      canvas.width = config.width * window.devicePixelRatio;
      canvas.height = config.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      canvas.style.width = `${config.width}px`;
      canvas.style.height = `${config.height}px`;
    };

    setCanvasSize();
    window.addEventListener('resize', setCanvasSize);

    let lastTime = 0;
    const targetFPS = 60;
    const frameInterval = 1000 / targetFPS;

    const animate = (currentTime: number) => {
      if (currentTime - lastTime < frameInterval) {
        animationRef.current = requestAnimationFrame(animate);
        return;
      }

      lastTime = currentTime;

      // Clear canvas
      ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
      ctx.fillRect(0, 0, config.width, config.height);

      // Update node activations
      setNodes(prevNodes => {
        const updatedNodes = prevNodes.map(node => {
          let newActivation = node.baseActivation;
          
          if (isActive) {
            // Simulate neural activity
            newActivation += Math.sin(currentTime * 0.005 + node.x * 0.01) * 0.3 * activityLevel;
            newActivation += Math.random() * 0.2 * activityLevel;
            
            // Propagate activation from previous layer
            if (node.layer > 0) {
              const prevLayerNodes = prevNodes.filter(n => n.layer === node.layer - 1);
              const inputSum = prevLayerNodes.reduce((sum, prevNode) => {
                const connection = connections.find(c => c.from === prevNode.id && c.to === node.id);
                return sum + (prevNode.activation * (connection?.weight || 0));
              }, 0);
              
              newActivation += Math.max(0, Math.min(1, inputSum * 0.1));
            }
          }

          return {
            ...node,
            activation: Math.max(0, Math.min(1, newActivation))
          };
        });

        return updatedNodes;
      });

      // Update connection activities
      setConnections(prevConnections => {
        return prevConnections.map(connection => {
          const fromNode = nodes.find(n => n.id === connection.from);
          const activity = isActive ? 
            (fromNode?.activation || 0) * Math.abs(connection.weight) * activityLevel :
            0;
          
          return {
            ...connection,
            activity: Math.max(0, Math.min(1, activity))
          };
        });
      });

      // Draw connections
      connections.forEach(connection => {
        const fromNode = nodes.find(n => n.id === connection.from);
        const toNode = nodes.find(n => n.id === connection.to);
        
        if (!fromNode || !toNode) return;

        const opacity = Math.max(0.1, connection.activity);
        const lineWidth = 1 + connection.activity * 2;
        
        // Connection color based on weight and activity
        const isPositive = connection.weight > 0;
        const baseColor = isPositive ? colors.connection : colors.pulse;
        
        ctx.strokeStyle = `${baseColor}${Math.floor(opacity * 255).toString(16).padStart(2, '0')}`;
        ctx.lineWidth = lineWidth;
        
        // Draw connection line
        ctx.beginPath();
        ctx.moveTo(fromNode.x, fromNode.y);
        ctx.lineTo(toNode.x, toNode.y);
        ctx.stroke();

        // Draw activity pulse
        if (connection.activity > 0.5 && isActive) {
          const pulseProgress = (currentTime * 0.01) % 1;
          const pulseX = fromNode.x + (toNode.x - fromNode.x) * pulseProgress;
          const pulseY = fromNode.y + (toNode.y - fromNode.y) * pulseProgress;
          
          ctx.fillStyle = colors.active;
          ctx.beginPath();
          ctx.arc(pulseX, pulseY, 2, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      // Draw nodes
      nodes.forEach(node => {
        const radius = 6 + node.activation * 4;
        const opacity = 0.3 + node.activation * 0.7;
        
        // Node glow
        if (node.activation > 0.5) {
          ctx.fillStyle = `${colors.pulse}30`;
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius + 6, 0, Math.PI * 2);
          ctx.fill();
        }
        
        // Node body
        ctx.fillStyle = `${colors.node}${Math.floor(opacity * 255).toString(16).padStart(2, '0')}`;
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.fill();
        
        // Node core
        if (node.activation > 0.3) {
          ctx.fillStyle = colors.active;
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius * 0.4, 0, Math.PI * 2);
          ctx.fill();
        }
        
        // Node border
        ctx.strokeStyle = `${colors.connection}80`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.stroke();
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener('resize', setCanvasSize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [nodes, connections, isActive, activityLevel, colors, config]);

  return (
    <div className={`relative ${className}`} style={{ width: config.width, height: config.height }}>
      <canvas
        ref={canvasRef}
        className="absolute inset-0"
        style={{
          width: config.width,
          height: config.height
        }}
      />
      
      {/* Layer labels */}
      <div className="absolute inset-0 pointer-events-none">
        {config.layers.map((_, layerIndex) => {
          const x = (layerIndex / (config.layers.length - 1)) * (config.width - 60) + 30;
          
          let label = 'Hidden';
          if (layerIndex === 0) label = 'Input';
          if (layerIndex === config.layers.length - 1) label = 'Output';
          
          return (
            <div
              key={layerIndex}
              className="absolute text-xs text-gray-400 font-mono transform -translate-x-1/2"
              style={{ 
                left: x, 
                bottom: -20,
                color: colors.connection
              }}
            >
              {label}
            </div>
          );
        })}
      </div>
      
      {/* Activity indicator */}
      <div className="absolute top-2 right-2 flex items-center space-x-2">
        <div 
          className={`w-2 h-2 rounded-full ${isActive ? 'animate-pulse' : 'opacity-30'}`}
          style={{ backgroundColor: colors.active }}
        />
        <span className="text-xs font-mono" style={{ color: colors.connection }}>
          {isActive ? 'ACTIVE' : 'IDLE'}
        </span>
      </div>
    </div>
  );
};

export default NeuralNetwork;