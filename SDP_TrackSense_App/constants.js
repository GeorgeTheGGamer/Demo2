// Shared network configuration for TrackSense app.
// Update LAPTOP_IP here and it will apply everywhere.
export const LAPTOP_IP = '192.168.118.173';
export const HTTP_BASE  = `http://${LAPTOP_IP}:5050`;
export const WS_URL     = `ws://${LAPTOP_IP}:5050/ws/status`;
