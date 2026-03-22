import { useState, useEffect, useRef } from 'react';
import * as Location from 'expo-location';

// Haversine formula to calculate distance between two coordinates in meters
function getDistance(lat1, lon1, lat2, lon2) {
  const R = 6371e3; // Earth radius in meters
  const p1 = lat1 * Math.PI / 180;
  const p2 = lat2 * Math.PI / 180;
  const dp = (lat2 - lat1) * Math.PI / 180;
  const dl = (lon2 - lon1) * Math.PI / 180;

  const a = Math.sin(dp / 2) * Math.sin(dp / 2) +
            Math.cos(p1) * Math.cos(p2) *
            Math.sin(dl / 2) * Math.sin(dl / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c; // distance in meters
}

export function useLocationTracking(isActive) {
  const [coordinates, setCoordinates] = useState([]);
  const [distance, setDistance] = useState(0); // overall distance in meters
  const [errorMsg, setErrorMsg] = useState(null);
  const [hasPermission, setHasPermission] = useState(false);
  const subscriptionRef = useRef(null);

  useEffect(() => {
    (async () => {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setErrorMsg('Permission to access location was denied');
        setHasPermission(false);
        return;
      }
      setHasPermission(true);
    })();
  }, []);

  useEffect(() => {
    if (!isActive || !hasPermission) {
      if (subscriptionRef.current) {
        subscriptionRef.current.remove();
        subscriptionRef.current = null;
      }
      return;
    }

    (async () => {
      // Start tracking
      try {
        const sub = await Location.watchPositionAsync(
          {
            accuracy: Location.Accuracy.High,
            timeInterval: 2000,
            distanceInterval: 5, // minimum 5 meters to record new point
          },
          (location) => {
            const { latitude, longitude, altitude, accuracy } = location.coords;

            // Ignore highly inaccurate GPS pings (radius > 20 meters)
            if (accuracy != null && accuracy > 20) {
              return;
            }

            const newPoint = {
              latitude,
              longitude,
              altitude,
              timestamp: location.timestamp,
            };

            setCoordinates((prev) => {
              if (prev.length > 0) {
                const lastPoint = prev[prev.length - 1];
                const dist = getDistance(
                  lastPoint.latitude,
                  lastPoint.longitude,
                  newPoint.latitude,
                  newPoint.longitude
                );

                // Ignore micro-movements under 5 meters to prevent stationary GPS jitter
                if (dist < 5) {
                  return prev;
                }

                // Update total distance immediately
                setDistance((d) => d + dist);
              }
              return [...prev, newPoint];
            });
          }
        );
        subscriptionRef.current = sub;
      } catch (err) {
        setErrorMsg('Failed to start location tracking');
        console.error(err);
      }
    })();

    return () => {
      if (subscriptionRef.current) {
        subscriptionRef.current.remove();
        subscriptionRef.current = null;
      }
    };
  }, [isActive, hasPermission]);

  return {
    coordinates,
    distance,
    errorMsg,
    hasPermission
  };
}
