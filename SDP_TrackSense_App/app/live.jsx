import { useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Animated, Easing, Pressable, ScrollView, Text, View } from 'react-native';
import { router, useLocalSearchParams } from 'expo-router';
import { useTTS } from '../hooks/useTTS';
import { useVoiceCommand } from '../hooks/useVoiceCommand';
import { VoiceListeningOverlay } from '../components/VoiceListeningOverlay';
import MapView, { Polyline } from 'react-native-maps';
import { useLocationTracking } from '../hooks/useLocationTracking';
import { uploadRunToStrava, isAuthenticated } from '../services/strava';

import { LAPTOP_IP, HTTP_BASE, WS_URL } from '../constants';

const initialStatus = {
  running: false,
  front: {
    robot_status: 'NORMAL',
    ANGLE: 'ANGLE=0.00',
    object_detection: { warning: [], danger: [] },
    stop_conditions: [],
  },
  rear: {
    status: 'No feet detected',
    object_detection: { warning: [], danger: [] },
    stop_conditions: [],
  },
};

export default function LiveScreen() {
  const { saveToStrava } = useLocalSearchParams();
  const [status, setStatus] = useState(initialStatus);
  const [connected, setConnected] = useState(false);
  const [isHoldStopping, setIsHoldStopping] = useState(false);
  const [autoStopWarning, setAutoStopWarning] = useState(null);
  const wsRef = useRef(null);
  const { speak } = useTTS();
  const { isListening, isSpeechDetected } = useVoiceCommand({
    alwaysListening: true,
    announceOnStart: false,
    onStopCommand: handleStop,
  });
  const announcedAutoStopReasonRef = useRef('None');
  const stopTriggeredRef = useRef(false);
  const holdProgress = useRef(new Animated.Value(0)).current;
  const prevFrontStopRef = useRef('None');
  const prevRearStopRef = useRef('None');
  const frontStopTimerRef = useRef(null);
  const rearStopTimerRef = useRef(null);
  const pointerTimerRef = useRef(null);

  const { coordinates, distance, hasPermission } = useLocationTracking(!stopTriggeredRef.current);
  const [startTime] = useState(Date.now());
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      if (!stopTriggeredRef.current) {
        setElapsedMs(Date.now() - startTime);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const formattedTime = new Date(elapsedMs).toISOString().substring(11, 19);
  const formattedDistance = (distance / 1000).toFixed(2);
  const pace = distance > 0 ? (elapsedMs / 1000 / 60) / (distance / 1000) : 0;
  const formattedPace = pace > 0 ? `${Math.floor(pace)}:${Math.floor((pace % 1) * 60).toString().padStart(2, '0')}` : '0:00';

  const frontWarningText = useMemo(() => (status.front.object_detection.warning || []).join(', ') || 'None', [status]);
  const frontDangerText = useMemo(() => (status.front.object_detection.danger || []).join(', ') || 'None', [status]);
  const frontStopText = useMemo(() => (status.front.stop_conditions || []).join(', ') || 'None', [status]);
  const rearWarningText = useMemo(() => (status.rear.object_detection.warning || []).join(', ') || 'None', [status]);
  const rearDangerText = useMemo(() => (status.rear.object_detection.danger || []).join(', ') || 'None', [status]);
  const rearStopText = useMemo(() => (status.rear.stop_conditions || []).join(', ') || 'None', [status]);

  useEffect(() => {
    speak('To stop, hold the screen for 3 seconds, or say Track Stop.');
  }, []);

  // Announce stop condition changes via TTS after 3 s of stability
  useEffect(() => {
    if (frontStopText === prevFrontStopRef.current) return;
    prevFrontStopRef.current = frontStopText;
    clearTimeout(frontStopTimerRef.current);
    if (frontStopText !== 'None') {
      frontStopTimerRef.current = setTimeout(() => {
        // Only speak if the condition is still the same after 3 s
        if (prevFrontStopRef.current === frontStopText) {
          speak(`Front: ${frontStopText}`);
        }
      }, 3000);
    }
  }, [frontStopText]);

  useEffect(() => {
    if (rearStopText === prevRearStopRef.current) return;
    prevRearStopRef.current = rearStopText;
    clearTimeout(rearStopTimerRef.current);
    if (rearStopText !== 'None') {
      rearStopTimerRef.current = setTimeout(() => {
        if (prevRearStopRef.current === rearStopText) {
          speak(`Rear: ${rearStopText}`);
        }
      }, 3000);
    }
  }, [rearStopText]);

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    let pollTimer = null;

    const fetchStatus = async () => {
      try {
        const res = await fetch(`${HTTP_BASE}/status`);
        if (!res.ok) return;
        const payload = await res.json();
        setStatus(payload);
      } catch (_) {
        // ignore poll failures; websocket may still be active
      }
    };

    ws.onopen = () => setConnected(true);

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        setStatus(payload);

        if (payload.auto_stop_reason && payload.auto_stop_reason.length > 0) {
          const reason = payload.auto_stop_reason.join(', ');
          if (announcedAutoStopReasonRef.current !== reason) {
            announcedAutoStopReasonRef.current = reason;
          }
          setAutoStopWarning(reason);
        } else {
          setAutoStopWarning(null);
          announcedAutoStopReasonRef.current = 'None';
        }
      } catch (err) {
        console.error(err);
      }
    };

    ws.onerror = (event) => {
      // Ignore offline errors in the UI
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
    };

    // Fallback consistency channel: poll status periodically
    pollTimer = setInterval(fetchStatus, 400);
    fetchStatus();

    return () => {
      if (pollTimer) clearInterval(pollTimer);
      ws.close();
    };
  }, []);

  async function handleStop() {
    if (stopTriggeredRef.current) return; // Prevent duplicate rapid executions
    const allStopConditions = [
      ...(status.front.stop_conditions || []),
      ...(status.rear.stop_conditions || []),
    ].filter(Boolean);
    const stopReason = autoStopWarning || Array.from(new Set(allStopConditions)).join(', ') || 'Stopped by user';

    stopTriggeredRef.current = true;
    setIsHoldStopping(true);
    speak('Stopping robot and finishing run');

    // Attempt Strava upload directly if saving is enabled
    try {
      const isStrava = await isAuthenticated();
      if (isStrava && coordinates.length > 0 && saveToStrava !== 'false') {
        await uploadRunToStrava(coordinates, elapsedMs, distance);
      }
    } catch (err) {
      console.error('Strava upload failed', err);
    }

    try {
      const response = await fetch(`${HTTP_BASE}/command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'STOP' }),
      });

      if (!response.ok) {
        Alert.alert('Error', 'Failed to send STOP command.');
        return;
      }

      if (wsRef.current) {
        wsRef.current.close();
      }
      router.replace({ pathname: '/', params: { stopReason } });
    } catch (error) {
      stopTriggeredRef.current = false;
      setIsHoldStopping(false);
      holdProgress.setValue(0);
      console.error(error);
      Alert.alert('Connection Failed', 'Could not reach the laptop.');
    }
  }



  function beginHoldToStop() {
    stopTriggeredRef.current = false;
    setIsHoldStopping(true);
    holdProgress.stopAnimation();
    holdProgress.setValue(0);
    Animated.timing(holdProgress, {
      toValue: 1,
      duration: 1000, // 1s visual + 2s delayPressIn = 3s total hold
      easing: Easing.linear,
      useNativeDriver: true,
    }).start();
  }

  function cancelHoldToStop() {
    if (stopTriggeredRef.current) {
      return;
    }

    setIsHoldStopping(false);
    holdProgress.stopAnimation();
    Animated.timing(holdProgress, {
      toValue: 0,
      duration: 180,
      easing: Easing.out(Easing.ease),
      useNativeDriver: true,
    }).start();
  }


  return (
    <View className="flex-1 bg-slate-950">
      <Animated.View
        pointerEvents="none"
        style={{ opacity: holdProgress, zIndex: 50 }}
        className="absolute inset-0 bg-orange-600/95 items-center justify-center p-8"
      >
        <Text className="text-4xl font-bold text-white text-center">Keep Holding...</Text>
        <Text className="text-xl font-semibold text-orange-200 text-center mt-4">Release to cancel</Text>
      </Animated.View>

      <ScrollView
        className="flex-1"
        showsVerticalScrollIndicator={false}
      >
        <Pressable
          style={{ minHeight: '100%', paddingTop: 56, paddingBottom: 120, paddingHorizontal: 20 }}
          delayPressIn={2000}
          delayLongPress={3000}
          onPressIn={beginHoldToStop}
          onPressOut={cancelHoldToStop}
          onLongPress={handleStop}
        >
          <View className="gap-6">
          <View className="px-1">
            <View className="flex-row items-center justify-between gap-4">
              <Text className="text-3xl font-bold tracking-tight text-white">Status</Text>
              <Text className={`rounded-full px-4 py-2 text-sm font-semibold ${connected ? 'bg-emerald-500/15 text-emerald-300' : 'bg-red-500/15 text-red-300'}`}>
                {connected ? 'Connected' : 'Disconnected'}
              </Text>
            </View>
            <Text className="mt-3 text-base leading-6 text-slate-300">
              Say TrackStop, or hold anywhere on the screen for 3 seconds to stop.
            </Text>
          </View>

          <View className="flex-row justify-between rounded-[32px] border border-slate-800 bg-slate-900 px-6 py-5">
            <View>
              <Text className="text-sm font-semibold text-slate-400">Time</Text>
              <Text className="text-2xl font-bold text-white tracking-widest">{formattedTime}</Text>
            </View>
            <View>
              <Text className="text-sm font-semibold text-slate-400">Distance</Text>
              <Text className="text-2xl font-bold text-white">{formattedDistance} km</Text>
            </View>
            <View>
              <Text className="text-sm font-semibold text-slate-400">Pace</Text>
              <Text className="text-2xl font-bold text-white">{formattedPace} /km</Text>
            </View>
          </View>

          <View pointerEvents="none" className="h-72 overflow-hidden rounded-[32px] border border-slate-800 bg-slate-900">
            {hasPermission && coordinates.length > 0 ? (
              <MapView 
                style={{ flex: 1, width: '100%', height: '100%' }}
                showsUserLocation
                followsUserLocation
                pitchEnabled={false}
                rotateEnabled={false}
                scrollEnabled={false}
                zoomEnabled={false}
                initialRegion={{
                  latitude: coordinates[0].latitude,
                  longitude: coordinates[0].longitude,
                  latitudeDelta: 0.005,
                  longitudeDelta: 0.005,
                }}
              >
                <Polyline 
                  coordinates={coordinates} 
                  strokeColor="#f97316" // Orange 500
                  strokeWidth={5} 
                />
              </MapView>
            ) : hasPermission ? (
              <View className="flex-1 items-center justify-center p-6">
                <Text className="text-center text-slate-400 font-medium tracking-wide">Acquiring GPS Signal...</Text>
              </View>
            ) : (
              <View className="flex-1 items-center justify-center p-6">
                <Text className="text-center text-slate-500">Enable location permissions to see the map</Text>
              </View>
            )}
          </View>



          <View className="rounded-[32px] border border-slate-800 bg-slate-900 p-6">
            <Text className="mb-4 text-3xl font-bold text-white">Front Camera</Text>
            <Text className="text-xl leading-8 text-slate-200">Robot Status: {status.front.robot_status}</Text>
            <Text className="mt-4 text-xl leading-8 text-amber-300">Warnings: {frontWarningText}</Text>
            <Text className="mt-2 text-xl leading-8 text-red-300">Dangers: {frontDangerText}</Text>
            <Text className="mt-2 text-xl leading-8 text-white">Stop Conditions: {frontStopText}</Text>
          </View>

          <View className="rounded-[32px] border border-slate-800 bg-slate-900 p-6">
            <Text className="mb-4 text-3xl font-bold text-white">Rear Camera</Text>
            <Text className="text-xl leading-8 text-slate-200">Foot Status: {status.rear.status}</Text>
            <Text className="mt-4 text-xl leading-8 text-amber-300">Warnings: {rearWarningText}</Text>
            <Text className="mt-2 text-xl leading-8 text-red-300">Dangers: {rearDangerText}</Text>
            <Text className="mt-2 text-xl leading-8 text-white">Stop Conditions: {rearStopText}</Text>
          </View>
        </View>
        </Pressable>
      </ScrollView>

      <VoiceListeningOverlay
        isListening={isSpeechDetected}
        idleText="Voice control is active. Say TrackStop, or hold anywhere for 3 seconds."
        listeningText="Speech detected. Listening for TrackStop."
      />
    </View>
  );
}
