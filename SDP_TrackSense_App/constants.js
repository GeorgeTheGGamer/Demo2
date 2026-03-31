export const LAPTOP_IP_LIST = ['192.168.118.173', '172.20.10.6'];

// Default to the first IP, will be updated automatically on launch
export let LAPTOP_IP = LAPTOP_IP_LIST[0]; 
export let HTTP_BASE = `http://${LAPTOP_IP}:5050`;
export let WS_URL = `ws://${LAPTOP_IP}:5050/ws/status`;

export const resolveActiveIP = async () => {
    for (const ip of LAPTOP_IP_LIST) {
        try {
            // Promise.race to enforce a 1500ms timeout for each IP
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 1500);
            
            const res = await fetch(`http://${ip}:5050/status`, { signal: controller.signal });
            clearTimeout(timeoutId);
            
            if (res.ok) {
                LAPTOP_IP = ip;
                HTTP_BASE = `http://${LAPTOP_IP}:5050`;
                WS_URL = `ws://${LAPTOP_IP}:5050/ws/status`;
                console.log('Actively resolved Laptop IP: ', LAPTOP_IP);
                return true;
            }
        } catch (e) {
            // Timeout or connection error, try the next
        }
    }
    return false;
};
