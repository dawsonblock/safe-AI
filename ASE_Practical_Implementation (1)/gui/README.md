# FDQC-Cockpit Production GUI

Modern web-based dashboard for monitoring and controlling the FDQC-Cockpit system.

## Features

### 🎯 Real-Time Monitoring
- System status and health metrics
- Validation statistics (approval rate, error rate, risk scores)
- Pattern memory utilization
- BERT performance metrics
- Live event stream with Server-Sent Events

### 🎮 Action Control
- Interactive action selection interface
- JSON-based observation input
- Multi-action selection
- Real-time safety validation
- Adversarial detection alerts

### 📊 Visualization
- Metrics dashboard with live updates
- Pattern memory progress bars
- Activity log with event history
- Alert monitoring
- Performance statistics

### 💾 System Management
- Manual checkpoint saving
- Configuration viewing
- Real-time event notifications
- Error tracking and alerts

## Quick Start

### 1. Install Dependencies

```bash
cd gui
pip install -r requirements.txt
```

### 2. Ensure Parent Configuration

Make sure `.env` file exists in parent directory with:
```bash
DEEPSEEK_API_KEY=your-api-key-here
```

### 3. Start the GUI

```bash
python app.py
```

### 4. Open Dashboard

Navigate to: **http://localhost:5000**

## Architecture

### Backend (Flask)
- **`app.py`**: Main Flask application
  - RESTful API endpoints
  - Server-Sent Events for real-time updates
  - System initialization and management
  - Integration with all FDQC components

### Frontend (HTML/CSS/JavaScript)
- **`templates/dashboard.html`**: Single-page dashboard
  - Responsive design
  - Real-time updates via EventSource
  - Interactive action selection
  - Metrics visualization

## API Endpoints

### System Status
```
GET /api/status
```
Returns system status, configuration, and metrics.

### Action Selection
```
POST /api/action/select
Body: {
  "observation": {...},
  "available_actions": [...]
}
```
Selects action with safety validation.

### Record Outcome
```
POST /api/action/record
Body: {
  "observation": {...},
  "action": "...",
  "reward": 1.0,
  "was_safe": true
}
```
Records action outcome for learning.

### Monitoring Metrics
```
GET /api/monitoring/metrics
```
Returns detailed monitoring metrics.

### Recent Alerts
```
GET /api/monitoring/alerts?severity=critical&limit=50
```
Returns recent system alerts.

### Save Checkpoint
```
POST /api/persistence/save
```
Saves system checkpoint.

### Event Stream
```
GET /api/events
```
Server-Sent Events stream for real-time updates.

### Configuration
```
GET /api/config
```
Returns current system configuration.

## Real-Time Events

The GUI subscribes to real-time events via Server-Sent Events:

- **`system_status`**: System status changes
- **`action_selected`**: Action selection events
- **`outcome_recorded`**: Outcome recording events
- **`checkpoint_saved`**: Checkpoint save events
- **`keepalive`**: Connection keepalive

## Dashboard Sections

### System Status Card
- Safety tier (GAMMA by default)
- Safe mode status
- BERT model information
- Device (CPU/GPU)

### Validation Metrics Card
- Total validations performed
- Approval rate percentage
- Error rate percentage
- Average risk score

### Pattern Memory Card
- Safe patterns count
- Unsafe patterns count
- Memory utilization percentage
- Visual progress bar

### Performance Card
- BERT encoding count
- Cache hit rate
- Average encoding time
- Simulation mode status

### Action Selection Panel
- JSON observation input
- Multi-action selection buttons
- Action execution button
- Checkpoint save button
- Result display with risk scores

### Recent Alerts
- Alert severity levels (info/warning/error/critical)
- Alert messages and metadata
- Timestamp information

### Activity Log
- Real-time event logging
- Action selections
- Outcome recordings
- Checkpoint saves
- Color-coded by event type

## Usage Examples

### Basic Action Selection

1. **Edit Observation** (optional):
   ```json
   {
     "current_file": "src/test.py",
     "task": "code_review",
     "complexity": 0.5
   }
   ```

2. **Select Actions**: Click action buttons to select/deselect

3. **Execute**: Click "Select Action" button

4. **View Results**: See risk score, approval status, and safety metrics

### Monitoring System Health

- **System Status Badge**: Green (ready), Yellow (initializing), Red (error)
- **Metrics**: Auto-update every 5 seconds
- **Pattern Memory**: Monitor utilization to prevent overflow
- **Performance**: Track BERT cache hit rate for optimization

### Saving Checkpoints

- Click "💾 Save Checkpoint" button
- Checkpoint saved to configured directory
- Event logged in activity log
- Confirmation in real-time updates

## Configuration

The GUI uses configuration from parent `.env` file:

```bash
# Safety Configuration
SAFETY_TIER=GAMMA
REQUIRE_HUMAN_APPROVAL=true
SAFE_MODE=true

# BERT Configuration
BERT_MODEL_NAME=bert-base-uncased
BERT_USE_GPU=true

# Monitoring Thresholds
ALERT_RISK_THRESHOLD=0.8
ALERT_ERROR_RATE_THRESHOLD=0.1
ALERT_MEMORY_UTILIZATION_THRESHOLD=0.9
```

## Troubleshooting

### GUI Won't Start

**Issue**: `ModuleNotFoundError`
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### System Shows "Initializing" Forever

**Issue**: Configuration error
**Solution**: Check `.env` file in parent directory
```bash
cd ..
cat .env  # Verify DEEPSEEK_API_KEY is set
```

### No Real-Time Updates

**Issue**: EventSource connection failed
**Solution**: Check browser console for errors, refresh page

### Action Selection Fails

**Issue**: System not initialized
**Solution**: Wait for status badge to show "ready"

### High Memory Usage

**Issue**: Pattern buffers full
**Solution**: Save checkpoint and restart system

## Development

### Project Structure
```
gui/
├── app.py                 # Flask backend
├── templates/
│   └── dashboard.html     # Frontend dashboard
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Adding New Features

1. **Backend**: Add route in `app.py`
2. **Frontend**: Add UI in `dashboard.html`
3. **Events**: Use `send_event()` for real-time updates
4. **API**: Follow RESTful conventions

### Customization

**Colors**: Edit CSS variables in `<style>` section
**Layout**: Modify grid layout in dashboard
**Metrics**: Add new metric cards as needed
**Events**: Subscribe to new event types

## Security

- **CORS**: Enabled for development (disable in production)
- **API Keys**: Never exposed to frontend
- **Input Validation**: JSON parsing with error handling
- **Rate Limiting**: Consider adding for production

## Performance

- **Auto-refresh**: 5-second interval (configurable)
- **Event Stream**: Efficient Server-Sent Events
- **Caching**: BERT embeddings cached automatically
- **Lazy Loading**: Metrics loaded on demand

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/events {
        proxy_pass http://localhost:5000;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
        proxy_buffering off;
        proxy_cache off;
    }
}
```

## Browser Compatibility

- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ⚠️ IE11 (not supported)

## License

See parent directory LICENSE file.

---

**🚀 GUI Ready for Production Use**

Start the server with `python app.py` and navigate to http://localhost:5000
