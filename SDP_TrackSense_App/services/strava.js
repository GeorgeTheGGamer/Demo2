import * as SecureStore from 'expo-secure-store';

export const STRAVA_CLIENT_ID = process.env.EXPO_PUBLIC_STRAVA_CLIENT_ID || '';
export const STRAVA_CLIENT_SECRET = process.env.EXPO_PUBLIC_STRAVA_CLIENT_SECRET || '';

// The generic Strava scopes needed to upload an activity
export const STRAVA_SCOPES = ['activity:write,read'];

// Auth endpoints
export const discovery = {
  authorizationEndpoint: 'https://www.strava.com/oauth/mobile/authorize',
  tokenEndpoint: 'https://www.strava.com/oauth/token',
  revocationEndpoint: 'https://www.strava.com/oauth/deauthorize',
};

// Keys for secure store
const TOKEN_KEY = 'tracksense_strava_access_token';
const REFRESH_KEY = 'tracksense_strava_refresh_token';
const EXPIRES_KEY = 'tracksense_strava_expires_at';

// Save tokens after login
export async function saveAuthTokens(authResponse) {
  if (!authResponse?.params?.code) return;
  
  // Exchange authorization code for token
  try {
    const res = await fetch(discovery.tokenEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        client_id: STRAVA_CLIENT_ID,
        client_secret: STRAVA_CLIENT_SECRET,
        code: authResponse.params.code,
        grant_type: 'authorization_code',
      }),
    });
    
    const data = await res.json();
    if (data.access_token) {
      await SecureStore.setItemAsync(TOKEN_KEY, data.access_token);
      await SecureStore.setItemAsync(REFRESH_KEY, data.refresh_token);
      await SecureStore.setItemAsync(EXPIRES_KEY, data.expires_at.toString());
    }
  } catch (error) {
    console.error("Failed to exchange code", error);
  }
}

// Get the current valid access token, optionally refreshing if expired
export async function getValidAccessToken() {
  const token = await SecureStore.getItemAsync(TOKEN_KEY);
  const expiresAt = await SecureStore.getItemAsync(EXPIRES_KEY);
  const refreshToken = await SecureStore.getItemAsync(REFRESH_KEY);

  if (!token) return null;

  // Check expiration (buffer of 60 seconds)
  if (expiresAt && Date.now() / 1000 > parseInt(expiresAt, 10) - 60) {
    if (!refreshToken) return null;

    // Refresh token
    try {
      const res = await fetch(discovery.tokenEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          client_id: STRAVA_CLIENT_ID,
          client_secret: STRAVA_CLIENT_SECRET,
          grant_type: 'refresh_token',
          refresh_token: refreshToken,
        }),
      });

      const data = await res.json();
      if (data.access_token) {
        await SecureStore.setItemAsync(TOKEN_KEY, data.access_token);
        await SecureStore.setItemAsync(REFRESH_KEY, data.refresh_token);
        await SecureStore.setItemAsync(EXPIRES_KEY, data.expires_at.toString());
        return data.access_token;
      }
    } catch (e) {
      console.error(e);
      return null;
    }
  }

  return token;
}

export async function clearAuth() {
  await SecureStore.deleteItemAsync(TOKEN_KEY);
  await SecureStore.deleteItemAsync(REFRESH_KEY);
  await SecureStore.deleteItemAsync(EXPIRES_KEY);
}

// Check if user is logged into Strava locally
export async function isAuthenticated() {
  const token = await getValidAccessToken();
  return !!token;
}

// Convert coordinates to GPX string
export function generateGPXString(coordinates) {
  let gpx = '<?xml version="1.0" encoding="UTF-8"?>\n';
  gpx += '<gpx creator="TrackSense" version="1.1" xmlns="http://www.topografix.com/GPX/1/1">\n';
  gpx += '  <trk>\n';
  gpx += '    <name>TrackSense Run</name>\n';
  gpx += '    <type>9</type>\n'; // 9 is run in Strava
  gpx += '    <trkseg>\n';
  
  for (const pt of coordinates) {
    gpx += `      <trkpt lat="${pt.latitude}" lon="${pt.longitude}">\n`;
    if (pt.altitude) gpx += `        <ele>${pt.altitude}</ele>\n`;
    if (pt.timestamp) {
      // Must be ISO 8601
      gpx += `        <time>${new Date(pt.timestamp).toISOString()}</time>\n`;
    }
    gpx += '      </trkpt>\n';
  }
  
  gpx += '    </trkseg>\n';
  gpx += '  </trk>\n';
  gpx += '</gpx>';
  return gpx;
}

// Upload the run to Strava using the GPX file
export async function uploadRunToStrava(coordinates, elapsedTimeMs, distanceMeters) {
  const token = await getValidAccessToken();
  if (!token) throw new Error("Not authenticated to Strava");
  if (!coordinates || coordinates.length === 0) throw new Error("No GPS coordinates to upload");

  const gpxString = generateGPXString(coordinates);
  
  // Prepare robust file approach using expo-file-system
  const FileSystem = require('expo-file-system/legacy');
  const fileUri = FileSystem.documentDirectory + 'run.gpx';
  await FileSystem.writeAsStringAsync(fileUri, gpxString, { encoding: 'utf8' });

  // Prepare multipart form data
  const formData = new FormData();
  formData.append('data_type', 'gpx');
  formData.append('activity_type', 'run');
  formData.append('name', 'TrackSense Run');
  formData.append('description', 'Recorded with TrackSense App.');
  formData.append('visibility', 'only_me');
  formData.append('file', {
    uri: fileUri,
    name: 'run.gpx',
    type: 'application/gpx+xml'
  });

  try {
    const response = await fetch('https://www.strava.com/api/v3/uploads', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`
      },
      body: formData,
    });
    
    if (!response.ok) {
      const err = await response.text();
      console.error("Strava Upload Error:", err);
      throw new Error("Failed to upload to Strava.");
    }
    
    return await response.json();
  } catch (err) {
    console.error(err);
    throw err;
  }
}
