# Technical Context Information

## Project Requirements
- GDPR compliant user management system
- Handle 10k+ concurrent users  
- PCI DSS Level 1 compliance required
- Tech Stack: Node.js, PostgreSQL, Redis, Docker
- Performance Target: <200ms response time, 99.9% uptime

## Security Requirements
- OAuth 2.0 authentication required
- Encrypted PII storage mandatory
- Audit logging for all operations
- Regular security scans and penetration testing

## Development Context
- Legacy system migration in progress
- Must maintain backward compatibility with v1 API
- Security audit scheduled for next month
- Team has limited experience with OAuth implementation

## Compliance Constraints
- SOC 2 Type II certification required
- GDPR data retention policies must be enforced
- PCI DSS requirements for payment data handling
- Regular compliance audits and reporting

## Performance Benchmarks
- Current system handles 2k concurrent users
- Database queries must complete in <50ms
- API endpoints must respond in <200ms
- 99.9% uptime SLA with customers
