#!/bin/bash
# Jarvis v2.0 Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-development}
COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="jarvis-v2"

echo -e "${BLUE}🚀 Jarvis v2.0 Deployment Script${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}Project: ${PROJECT_NAME}${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_status "Docker and Docker Compose are installed"
}

# Create necessary directories
setup_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p logs/nginx
    mkdir -p logs/jarvis-core
    mkdir -p logs/jarvis-audio
    mkdir -p logs/jarvis-web
    mkdir -p config/nginx/ssl
    mkdir -p config/grafana/provisioning
    mkdir -p data/mongodb
    mkdir -p data/redis
    mkdir -p data/prometheus
    mkdir -p data/grafana
    
    print_status "Directories created successfully"
}

# Set up environment variables
setup_environment() {
    print_status "Setting up environment variables..."
    
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating default .env file..."
        
        cat > .env << EOF
# Jarvis v2.0 Environment Configuration
ENVIRONMENT=${ENVIRONMENT}
PROJECT_NAME=${PROJECT_NAME}

# Database Configuration
REDIS_URL=redis://redis:6379
MONGODB_URL=mongodb://jarvis:jarvis_secure_password@mongodb:27017/jarvis_v2

# Service URLs
CORE_SERVICE_URL=http://jarvis-core:8000
AUDIO_SERVICE_URL=http://jarvis-audio:8001
WEB_SERVICE_URL=http://jarvis-web:3000

# Security
JWT_SECRET=your_jwt_secret_here_change_in_production
ADMIN_PASSWORD=admin_password_change_in_production

# External APIs
OPENWEATHER_API_KEY=your_openweathermap_api_key_here

# Monitoring
PROMETHEUS_RETENTION=15d
GRAFANA_ADMIN_PASSWORD=admin

# Docker
COMPOSE_PROJECT_NAME=${PROJECT_NAME}
EOF
        
        print_warning "Please edit .env file with your actual configuration values"
    fi
    
    print_status "Environment variables configured"
}

# Build and start services
deploy_services() {
    print_status "Building and starting services..."
    
    # Stop existing services
    docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} down
    
    # Build images
    print_status "Building Docker images..."
    docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} build
    
    # Start services
    print_status "Starting services..."
    docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} up -d
    
    print_status "Services started successfully"
}

# Health check
health_check() {
    print_status "Performing health checks..."
    
    # Wait for services to start
    sleep 30
    
    # Check core service
    if curl -f http://localhost:8000/api/v2/health > /dev/null 2>&1; then
        print_status "✅ Core service is healthy"
    else
        print_error "❌ Core service health check failed"
    fi
    
    # Check audio service
    if curl -f http://localhost:8001/api/v2/audio/health > /dev/null 2>&1; then
        print_status "✅ Audio service is healthy"
    else
        print_error "❌ Audio service health check failed"
    fi
    
    # Check web service
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        print_status "✅ Web service is healthy"
    else
        print_error "❌ Web service health check failed"
    fi
    
    # Check nginx
    if curl -f http://localhost:80/health > /dev/null 2>&1; then
        print_status "✅ Nginx is healthy"
    else
        print_error "❌ Nginx health check failed"
    fi
    
    print_status "Health checks completed"
}

# Show deployment status
show_status() {
    echo ""
    echo -e "${GREEN}🎉 Jarvis v2.0 Deployment Status${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    
    docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} ps
    
    echo ""
    echo -e "${GREEN}Service URLs:${NC}"
    echo -e "${BLUE}• Web Interface:${NC} http://localhost:3000"
    echo -e "${BLUE}• Core API:${NC} http://localhost:8000"
    echo -e "${BLUE}• Audio API:${NC} http://localhost:8001"
    echo -e "${BLUE}• Nginx:${NC} http://localhost:80"
    echo -e "${BLUE}• Prometheus:${NC} http://localhost:9090"
    echo -e "${BLUE}• Grafana:${NC} http://localhost:3001"
    echo ""
    
    echo -e "${GREEN}WebSocket URLs:${NC}"
    echo -e "${BLUE}• Core WebSocket:${NC} ws://localhost:8000/ws/{user_id}"
    echo -e "${BLUE}• Audio WebSocket:${NC} ws://localhost:8001/ws/audio/{user_id}"
    echo ""
    
    echo -e "${GREEN}Commands:${NC}"
    echo -e "${BLUE}• View logs:${NC} docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} logs -f"
    echo -e "${BLUE}• Stop services:${NC} docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} down"
    echo -e "${BLUE}• Restart services:${NC} docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} restart"
    echo ""
}

# Main deployment flow
main() {
    print_status "Starting Jarvis v2.0 deployment..."
    
    check_docker
    setup_directories
    setup_environment
    deploy_services
    health_check
    show_status
    
    echo -e "${GREEN}✅ Jarvis v2.0 deployment completed successfully!${NC}"
    echo -e "${BLUE}🚀 Your Jarvis assistant is now running at http://localhost:3000${NC}"
}

# Handle script arguments
case "$1" in
    "start"|"deploy"|"")
        main
        ;;
    "stop")
        print_status "Stopping Jarvis v2.0 services..."
        docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} down
        print_status "Services stopped"
        ;;
    "restart")
        print_status "Restarting Jarvis v2.0 services..."
        docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} restart
        print_status "Services restarted"
        ;;
    "logs")
        docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} logs -f
        ;;
    "status")
        docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} ps
        ;;
    "clean")
        print_status "Cleaning up Jarvis v2.0 deployment..."
        docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} down -v
        docker system prune -f
        print_status "Cleanup completed"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status|clean}"
        echo ""
        echo "Commands:"
        echo "  start/deploy  - Deploy Jarvis v2.0 (default)"
        echo "  stop         - Stop all services"
        echo "  restart      - Restart all services"
        echo "  logs         - View service logs"
        echo "  status       - Show service status"
        echo "  clean        - Clean up deployment"
        exit 1
        ;;
esac