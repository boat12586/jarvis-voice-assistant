# Jarvis v2.0 Production Deployment Guide

## 🚀 Overview

This guide provides comprehensive instructions for deploying Jarvis Voice Assistant v2.0 in a production environment with high availability, security, and scalability.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Ubuntu 20.04 LTS or newer, CentOS 8+, or Docker-compatible system
- **RAM**: 8GB minimum (16GB recommended)
- **CPU**: 4 cores minimum (8 cores recommended)
- **Storage**: 50GB minimum (100GB recommended)
- **Network**: Stable internet connection with open ports 80, 443, 8000, 8001

### Software Dependencies
- **Docker**: Version 20.10 or newer
- **Docker Compose**: Version 2.0 or newer
- **Git**: For source code management
- **OpenSSL**: For SSL certificate generation
- **Curl**: For health checks and testing

## 🔧 Quick Start Deployment

### 1. Clone and Setup
```bash
# Clone repository
git clone https://github.com/your-org/jarvis-voice-assistant.git
cd jarvis-voice-assistant

# Make deployment script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

### 2. Access Services
After deployment, access your services at:
- **Web Interface**: http://localhost:3000
- **Core API**: http://localhost:8000
- **Audio API**: http://localhost:8001
- **Monitoring**: http://localhost:9090 (Prometheus), http://localhost:3001 (Grafana)

## 🏗️ Architecture Overview

### Service Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │    │   Web Interface │    │   Monitoring    │
│   (Port 80/443) │    │   (Port 3000)   │    │   (Port 9090)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
┌─────────────────┐                               ┌─────────────────┐
│   Core Service  │                               │  Audio Service  │
│   (Port 8000)   │                               │   (Port 8001)   │
└─────────────────┘                               └─────────────────┘
         │                                                 │
         └────────────────────────┬────────────────────────┘
                                  │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      Redis      │    │    MongoDB      │    │    Plugins      │
│   (Port 6379)   │    │  (Port 27017)   │    │    System       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow
1. **User Request** → Nginx → Web Interface
2. **API Calls** → Nginx → Core Service
3. **Voice Processing** → Audio Service
4. **Plugin Commands** → Plugin System
5. **Session Data** → Redis Cache
6. **Persistent Data** → MongoDB

## 🔐 Security Configuration

### 1. SSL/TLS Setup
```bash
# Generate SSL certificates (use Let's Encrypt for production)
mkdir -p config/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout config/nginx/ssl/key.pem \
  -out config/nginx/ssl/cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=yourdomain.com"
```

### 2. Environment Variables
Create and configure `.env` file:
```env
# Production Environment
ENVIRONMENT=production
PROJECT_NAME=jarvis-v2

# Security
JWT_SECRET=your_super_secure_jwt_secret_here
ADMIN_PASSWORD=your_secure_admin_password
REDIS_PASSWORD=your_redis_password
MONGODB_PASSWORD=your_mongodb_password

# Database Configuration
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379
MONGODB_URL=mongodb://jarvis:${MONGODB_PASSWORD}@mongodb:27017/jarvis_v2

# External APIs
OPENWEATHER_API_KEY=your_openweathermap_api_key
GEMINI_API_KEY=your_gemini_api_key

# Domain Configuration
DOMAIN=yourdomain.com
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem

# Monitoring
PROMETHEUS_RETENTION=30d
GRAFANA_ADMIN_PASSWORD=secure_grafana_password
```

### 3. Firewall Configuration
```bash
# Ubuntu/Debian
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw enable

# CentOS/RHEL
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

## 🎛️ Configuration Management

### 1. Docker Compose Override
Create `docker-compose.prod.yml`:
```yaml
version: '3.8'

services:
  jarvis-core:
    environment:
      - WORKERS=8
      - LOG_LEVEL=warning
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'
    restart: always

  jarvis-audio:
    environment:
      - WORKERS=4
      - LOG_LEVEL=warning
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '4'
    restart: always

  redis:
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1'

  mongodb:
    environment:
      - MONGO_INITDB_ROOT_PASSWORD=${MONGODB_PASSWORD}
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'
```

### 2. Production Deployment Command
```bash
# Deploy with production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 📊 Monitoring and Logging

### 1. Prometheus Configuration
The system includes built-in Prometheus monitoring:
- **Metrics Collection**: All services export metrics
- **Alerting**: Configure alerts for system health
- **Retention**: Default 30 days (configurable)

### 2. Grafana Dashboards
Access Grafana at `http://localhost:3001`:
- **Username**: admin
- **Password**: Set in environment variables
- **Dashboards**: Pre-configured for Jarvis metrics

### 3. Log Management
```bash
# View service logs
docker-compose logs -f jarvis-core
docker-compose logs -f jarvis-audio
docker-compose logs -f jarvis-web

# Log rotation (add to crontab)
0 2 * * * /usr/sbin/logrotate -f /etc/logrotate.d/jarvis
```

## 🔄 Backup and Recovery

### 1. Database Backup
```bash
# MongoDB backup
docker exec jarvis-mongodb mongodump --out /backup/mongodb/$(date +%Y%m%d_%H%M%S)

# Redis backup
docker exec jarvis-redis redis-cli BGSAVE
```

### 2. Configuration Backup
```bash
# Backup configuration files
tar -czf jarvis-config-$(date +%Y%m%d_%H%M%S).tar.gz config/ .env docker-compose.yml
```

### 3. Automated Backup Script
```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backup/jarvis"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR/$DATE

# Database backups
docker exec jarvis-mongodb mongodump --out $BACKUP_DIR/$DATE/mongodb
docker exec jarvis-redis redis-cli BGSAVE

# Configuration backup
tar -czf $BACKUP_DIR/$DATE/config.tar.gz config/ .env docker-compose.yml

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -type d -mtime +7 -exec rm -rf {} \;
```

## 🚀 Performance Optimization

### 1. Resource Allocation
```yaml
# Recommended resource allocation
services:
  jarvis-core:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'
        reservations:
          memory: 1G
          cpus: '1'

  jarvis-audio:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '4'
        reservations:
          memory: 2G
          cpus: '2'
```

### 2. Nginx Optimization
```nginx
# nginx.conf optimization
worker_processes auto;
worker_connections 4096;

# Enable compression
gzip on;
gzip_types text/plain application/json application/javascript text/css;

# Cache static files
location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

### 3. Database Optimization
```javascript
// MongoDB indexes
db.users.createIndex({ "user_id": 1 }, { unique: true });
db.sessions.createIndex({ "session_id": 1 }, { unique: true });
db.conversations.createIndex({ "timestamp": 1 });
```

## 📱 Mobile API Integration

### 1. Mobile-specific Endpoints
```bash
# Mobile authentication
POST /api/v2/mobile/auth/login
POST /api/v2/mobile/auth/refresh

# Mobile voice processing
POST /api/v2/mobile/voice/upload
GET /api/v2/mobile/voice/stream/{session_id}

# Mobile push notifications
POST /api/v2/mobile/notifications/register
POST /api/v2/mobile/notifications/send
```

### 2. Mobile Configuration
```yaml
# Add to docker-compose.yml
mobile-gateway:
  build: ./services/mobile-gateway
  ports:
    - "8002:8002"
  environment:
    - CORE_SERVICE_URL=http://jarvis-core:8000
    - AUDIO_SERVICE_URL=http://jarvis-audio:8001
    - PUSH_NOTIFICATION_KEY=${PUSH_NOTIFICATION_KEY}
```

## 🌐 Cloud Deployment

### 1. AWS Deployment
```bash
# Using AWS ECS
aws ecs create-cluster --cluster-name jarvis-v2
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs create-service --cluster jarvis-v2 --service-name jarvis-core --task-definition jarvis-core:1
```

### 2. Google Cloud Deployment
```bash
# Using Google Cloud Run
gcloud run deploy jarvis-core --image gcr.io/your-project/jarvis-core --platform managed
gcloud run deploy jarvis-audio --image gcr.io/your-project/jarvis-audio --platform managed
```

### 3. Azure Deployment
```bash
# Using Azure Container Instances
az container create --resource-group jarvis-rg --name jarvis-core --image your-registry/jarvis-core:latest
az container create --resource-group jarvis-rg --name jarvis-audio --image your-registry/jarvis-audio:latest
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Service Not Starting
```bash
# Check logs
docker-compose logs jarvis-core
docker-compose logs jarvis-audio

# Check resource usage
docker stats

# Restart services
docker-compose restart
```

#### 2. Database Connection Issues
```bash
# Check Redis connection
docker exec jarvis-redis redis-cli ping

# Check MongoDB connection
docker exec jarvis-mongodb mongo --eval "db.adminCommand('ping')"
```

#### 3. Audio Processing Issues
```bash
# Check audio service logs
docker-compose logs jarvis-audio

# Test audio endpoint
curl -X POST http://localhost:8001/api/v2/audio/health
```

### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check system metrics
curl http://localhost:9090/metrics

# Analyze logs
docker-compose logs --tail=1000 jarvis-core | grep ERROR
```

## 📋 Maintenance Tasks

### Daily Tasks
- [ ] Check service health status
- [ ] Monitor resource usage
- [ ] Review error logs
- [ ] Verify backup completion

### Weekly Tasks
- [ ] Update security patches
- [ ] Review performance metrics
- [ ] Clean up old logs
- [ ] Test disaster recovery

### Monthly Tasks
- [ ] Update dependencies
- [ ] Review and rotate secrets
- [ ] Performance optimization
- [ ] Capacity planning

## 🎯 Success Metrics

### Key Performance Indicators
- **System Uptime**: Target 99.9%
- **Response Time**: API < 200ms, Audio < 500ms
- **Error Rate**: < 0.1%
- **Concurrent Users**: Support 1000+ users
- **Audio Processing**: < 1s latency

### Monitoring Alerts
- Service health checks
- Resource usage thresholds
- Error rate spikes
- Response time degradation

## 🎉 Conclusion

This production deployment guide provides a comprehensive framework for deploying Jarvis v2.0 in a scalable, secure, and maintainable manner. The system supports:

✅ **High Availability**: Load balancing and failover  
✅ **Security**: SSL/TLS, authentication, authorization  
✅ **Scalability**: Horizontal scaling and resource optimization  
✅ **Monitoring**: Comprehensive metrics and alerting  
✅ **Backup**: Automated backup and recovery procedures  
✅ **Performance**: Optimized for high-throughput scenarios  

For additional support or custom deployment requirements, refer to the detailed documentation in each service directory.

---

**Last Updated**: July 18, 2025  
**Version**: 2.0.0  
**Status**: Production Ready  
**Support**: [Documentation](docs/) | [Issues](https://github.com/your-org/jarvis-voice-assistant/issues)