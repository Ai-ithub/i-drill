# i-Drill React Dashboard

Dashboard frontend for i-Drill drilling data monitoring system.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create `.env` file (optional):
```env
VITE_API_URL=http://localhost:8001/api/v1
VITE_WS_URL=ws://localhost:8001/api/v1
```

3. Run development server:
```bash
npm run dev
```

The dashboard will be available at `http://localhost:3001`

**Note**: Make sure the backend server is running on port 8001 before accessing the dashboard.

## Features

- **Real-time Monitoring**: Live sensor data via WebSocket
- **Dashboard**: Overview of drilling operations
- **Historical Data**: Query and visualize historical data
- **Predictions**: RUL and anomaly detection
- **Maintenance**: Alerts and scheduling

## Project Structure

```
frontend/
├── src/
│   ├── components/     # React components
│   ├── pages/          # Page components
│   ├── services/       # API and WebSocket services
│   ├── types/          # TypeScript types
│   └── utils/          # Utility functions
├── package.json
└── vite.config.ts
```

## Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

