# Jarvis v2.0 Security Configuration

This directory contains comprehensive security configurations for Jarvis v2.0 cloud deployment.

## Overview

The security system implements multiple layers of protection including:

- **Authentication & Authorization**: JWT-based authentication with OAuth2 integration
- **Role-Based Access Control (RBAC)**: Fine-grained permissions for different user roles
- **Network Security**: Network policies and ingress protection
- **Certificate Management**: Automated SSL/TLS certificate provisioning
- **Secret Management**: Vault integration for secure secret storage
- **Security Scanning**: Runtime security monitoring and vulnerability scanning
- **Audit Logging**: Comprehensive audit trails and compliance reporting

## Security Components

### 1. Authentication & Authorization (`auth-middleware.yaml`)

**JWT Authentication Service**
- Stateless JWT token validation
- Refresh token rotation
- Session management with Redis
- Brute force protection
- Rate limiting per user/IP

**OAuth2 Integration**
- Google OAuth2 provider
- Social login support
- OpenID Connect compatibility
- Multi-factor authentication ready

**Features:**
- Session timeout: 1 hour
- Refresh token: 24 hours
- Max sessions per user: 5
- Brute force protection: 5 attempts / 15 minutes
- Rate limiting: 100 requests / 15 minutes

### 2. Role-Based Access Control (`rbac.yaml`)

**Service Accounts:**
- `jarvis-core-service-account`: Core application permissions
- `jarvis-admin-service-account`: Administrative access
- `jarvis-user-service-account`: Standard user permissions
- `jarvis-readonly-service-account`: Read-only access
- `jarvis-monitoring-service-account`: Monitoring system access

**Roles:**
- **Admin**: Full access to all resources
- **User**: Access to application features and logs
- **ReadOnly**: View-only access to resources
- **Monitoring**: Metrics and monitoring access

**Cluster-Level Permissions:**
- Pod and service discovery
- Ingress and networking access
- Metrics collection
- Event creation for audit trails

### 3. Network Security (`security-policies.yaml`)

**Pod Security Policies:**
- Non-root execution required
- Privilege escalation disabled
- Read-only root filesystem
- Limited capabilities (drop ALL)
- Resource quotas and limits

**Network Policies:**
- Default deny all traffic
- Explicit allow rules for required communication
- Ingress isolation between namespaces
- Egress restrictions for external access
- Monitoring namespace isolation

**Resource Management:**
- CPU limits: 50m-2000m
- Memory limits: 64Mi-4Gi
- Storage limits: 1Gi-200Gi
- Pod limits: 50 per namespace

### 4. OAuth2 Proxy (`oauth2-proxy.yaml`)

**Authentication Gateway:**
- Google OAuth2 integration
- Cookie-based session management
- Reverse proxy authentication
- Domain-wide authentication
- Skip authentication for public endpoints

**Configuration:**
- Cookie expiration: 24 hours
- Cookie refresh: 1 hour
- Secure cookie settings
- HTTPS-only operation
- Cross-domain cookie support

**Endpoints:**
- Public: `/health`, `/metrics`, `/api/v2/public/*`
- Protected: All other endpoints
- WebSocket support: `/ws/*`

### 5. Vault Integration (`vault-integration.yaml`)

**Secret Management:**
- Kubernetes authentication
- Dynamic secret rotation
- Template-based secret injection
- Automatic unsealing
- Audit logging

**Vault Policies:**
- `jarvis-v2-policy`: Read access to Jarvis secrets
- Token lifecycle management
- Service account binding
- Namespace-based access control

**Secret Categories:**
- Database credentials
- API keys (OpenWeather, Gemini, FCM, APNS)
- JWT signing keys
- SSL/TLS certificates
- OAuth2 credentials

### 6. Certificate Management (`cert-manager.yaml`)

**Automated SSL/TLS:**
- Let's Encrypt integration
- DNS-01 and HTTP-01 challenges
- Google Cloud DNS integration
- Automatic certificate renewal
- Internal CA for service-to-service communication

**Certificate Types:**
- **External**: Let's Encrypt for public domains
- **Internal**: Self-signed CA for internal services
- **Monitoring**: Separate certificates for monitoring stack

**Domains Covered:**
- `jarvis.yourdomain.com` (main application)
- `api.jarvis.yourdomain.com` (API gateway)
- `audio.jarvis.yourdomain.com` (audio service)
- `mobile.jarvis.yourdomain.com` (mobile API)
- `auth.jarvis.yourdomain.com` (authentication)
- `monitoring.jarvis.yourdomain.com` (monitoring)
- `grafana.jarvis.yourdomain.com` (Grafana)
- `prometheus.jarvis.yourdomain.com` (Prometheus)

### 7. Security Scanning (`security-scanner.yaml`)

**Falco Runtime Security:**
- Kubernetes audit log analysis
- System call monitoring
- Container runtime security
- Custom rules for Jarvis workloads
- Real-time threat detection

**Trivy Vulnerability Scanner:**
- Container image scanning
- Configuration audit
- RBAC assessment
- Infrastructure assessment
- Compliance reporting

**Custom Security Rules:**
- Unauthorized network connections
- Privilege escalation attempts
- Sensitive file access
- Crypto mining detection
- Container escape attempts
- Malicious binary execution
- API anomaly detection
- Data exfiltration detection

## Security Best Practices

### 1. Authentication
- Use strong JWT secrets (64+ characters)
- Implement refresh token rotation
- Enable multi-factor authentication
- Use OAuth2 for external authentication
- Implement session timeouts

### 2. Authorization
- Follow principle of least privilege
- Use service accounts for applications
- Implement role-based access control
- Regular access reviews
- Audit permission changes

### 3. Network Security
- Default deny network policies
- Explicit allow rules only
- Ingress protection with rate limiting
- TLS for all communications
- Service mesh for internal traffic

### 4. Secret Management
- Use Vault for secret storage
- Rotate secrets regularly
- Never store secrets in code
- Use service account annotations
- Implement secret scanning

### 5. Monitoring & Compliance
- Enable audit logging
- Monitor security events
- Regular vulnerability scanning
- Compliance reporting
- Incident response procedures

## Deployment Instructions

### Prerequisites
1. Kubernetes cluster with RBAC enabled
2. Cert-manager installed
3. Ingress controller deployed
4. Vault server running
5. Prometheus operator installed

### Deployment Order
1. **RBAC and Security Policies**
   ```bash
   kubectl apply -f rbac.yaml
   kubectl apply -f security-policies.yaml
   ```

2. **Certificate Management**
   ```bash
   kubectl apply -f cert-manager.yaml
   ```

3. **Vault Integration**
   ```bash
   kubectl apply -f vault-integration.yaml
   ```

4. **OAuth2 Proxy**
   ```bash
   kubectl apply -f oauth2-proxy.yaml
   ```

5. **Authentication Middleware**
   ```bash
   kubectl apply -f auth-middleware.yaml
   ```

6. **Security Scanning**
   ```bash
   kubectl apply -f security-scanner.yaml
   ```

### Configuration Updates Required

1. **Domain Names**: Replace `yourdomain.com` with your actual domain
2. **Project ID**: Update GCP project ID in all configurations
3. **Secrets**: Update all placeholder secrets with actual values
4. **OAuth2**: Configure Google OAuth2 credentials
5. **Vault**: Initialize Vault with proper policies and secrets
6. **Monitoring**: Configure webhook URLs for security alerts

## Security Monitoring

### Metrics Collected
- Authentication attempts and failures
- Authorization decisions
- Network policy violations
- Certificate expiry status
- Vulnerability scan results
- Security event counts

### Alerts Configured
- Failed login attempts > 10/min
- Privilege escalation attempts
- Unauthorized network connections
- Certificate expiry < 30 days
- Critical vulnerabilities detected
- Security rule violations

### Audit Logs
- All authentication events
- Authorization decisions
- Administrative actions
- Security policy changes
- Certificate operations
- Vault access attempts

## Compliance

This security configuration supports compliance with:
- **SOC 2 Type 2**: Access controls and monitoring
- **ISO 27001**: Information security management
- **GDPR**: Data protection and privacy
- **HIPAA**: Healthcare data protection (when applicable)
- **PCI DSS**: Payment card industry standards

## Troubleshooting

### Common Issues
1. **Authentication failures**: Check JWT secret and OAuth2 configuration
2. **Certificate issues**: Verify DNS configuration and Let's Encrypt limits
3. **Network policies**: Review ingress/egress rules and pod labels
4. **Vault access**: Check service account bindings and policies
5. **Security scanner**: Verify RBAC permissions and node access

### Debug Commands
```bash
# Check authentication service
kubectl logs -n jarvis-v2 -l app=jarvis-auth-middleware

# Verify certificates
kubectl get certificates -n jarvis-v2

# Check security events
kubectl logs -n jarvis-v2-monitoring -l app=falco

# Review network policies
kubectl get networkpolicies -n jarvis-v2

# Check Vault status
kubectl logs -n jarvis-v2 -l app=vault-unsealer
```

## Security Contact

For security issues or questions:
- Email: security@jarvis.yourdomain.com
- Slack: #security-alerts
- Emergency: security-emergency@jarvis.yourdomain.com

## Updates and Maintenance

- **Weekly**: Review security alerts and audit logs
- **Monthly**: Update container images and security policies
- **Quarterly**: Penetration testing and security assessment
- **Annually**: Full security audit and compliance review