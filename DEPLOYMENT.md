# Deployment Guide

## Quick Start

### 1. Local Development
```bash
# Make run script executable
chmod +x run.sh

# Start the application (defaults to 3000)
PORT=3000 ./run.sh
```

The API will be available at `http://localhost:3000`

### 2. Docker Deployment
```bash
# Build and run
docker build -t data-analyst-agent .
```bash
docker run -e PORT=3000 -p 3000:3000 data-analyst-agent
```
```

### 3. Production Deployment Options

#### Heroku
```bash
# Install Heroku CLI and login
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main
```

#### DigitalOcean App Platform
1. Connect your GitHub repository
2. Select "Docker" as the source type
3. Set environment variables if needed
4. Deploy

#### AWS ECS/Fargate
1. Push image to ECR
2. Create ECS task definition
3. Create service with load balancer

#### Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway deploy
```

### 4. Environment Configuration

Set these environment variables in production:
- `PORT`: Server port (default: 3000)
- `DEBUG`: Set to "false" in production

### 5. Testing Your Deployment

Use the provided test script:
```bash
PORT=3000 python test_api.py
```

Or test manually:
```bash
# Health check
```bash
curl http://localhost:3000/health
```

```bash
curl "http://localhost:3000/api/" \

# API test
curl "http://localhost:8002/api/" \
  -F "questions.txt=@sample_questions.txt"
```

### 6. Performance Tuning

For production:
- Use multiple worker processes (gunicorn)
- Set up reverse proxy (nginx)
- Enable request logging
- Configure monitoring

Example with gunicorn:
```bash
pip install gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:3000 app:app
```
```

### 7. Monitoring

Check logs for:
- Request processing times
- Error rates
- Memory usage
- API response times

### 8. Security Considerations

- Set file upload limits
- Validate input files
- Use HTTPS in production
- Consider rate limiting
- Monitor for abuse
