import { useEffect, useState } from 'react';
import { router, useLocalSearchParams } from 'expo-router';
import { Alert, Pressable, ScrollView, Text, TouchableOpacity, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useTTS } from '../hooks/useTTS';
import { useVoiceCommand } from '../hooks/useVoiceCommand';
import { LAPTOP_IP, HTTP_BASE } from '../constants';
import { VoiceListeningOverlay } from '../components/VoiceListeningOverlay';

export default function App() {
  const { stopReason } = useLocalSearchParams();
  const insets = useSafeAreaInsets();
  const [banner, setBanner] = useState(stopReason || null);
  const { speak } = useTTS();
  const { isListening, isSpeechDetected } = useVoiceCommand({
    alwaysListening: true,
    announceOnStart: false,
    onStartCommand: handleStart,
    onStopCommand: handleStop,
  });

  // Clear banner param from URL after reading so it doesn't persist on re-render
  useEffect(() => {
    if (stopReason) {
      setBanner(stopReason);
      router.setParams({ stopReason: undefined, suppressSpeech: undefined });
    }
  }, [stopReason]);


  async function handleStart() {
    const url = `${HTTP_BASE}/command`;
    speak('Starting robot');
    try {
      // Send the POST request to your Laptop's Flask server
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'START' }),
      });

      if (response.ok) {
        router.replace('/live');
      } else {
        Alert.alert("Error", "Laptop received the request but something went wrong.");
      }
    } catch (error) {
      // This triggers if your laptop is offline, on a different Wi-Fi, or the IP is wrong
      console.error(error);
      Alert.alert("Connection Failed", "Could not reach the laptop at " + LAPTOP_IP);
    }
  }

  async function handleStop() {
    const url = `${HTTP_BASE}/command`;
    speak('Stopping robot');
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'STOP' }),
      });

      if (response.ok) {
        router.replace('/');
      }
    } catch (error) {
      console.error(error);
      Alert.alert("Connection Failed", "Could not reach the laptop.");
    }
  }

  return (
    <Pressable
      className="flex-1 bg-slate-950"
      onPress={handleStart}
    >

      {banner && (
        <View
          style={{ top: insets.top + 8 }}
          className="absolute left-4 right-4 z-10 rounded-3xl border border-red-400/40 bg-red-500/90 px-6 py-5 shadow-lg"
        >
          <Text className="mb-1 text-xl font-bold text-white">Robot Stopped</Text>
          <Text className="text-base text-red-50">{banner}</Text>
          <TouchableOpacity
            onPress={(event) => {
              event.stopPropagation();
              setBanner(null);
            }}
            className="mt-3 self-start rounded-full bg-white/15 px-4 py-2"
          >
            <Text className="text-sm font-semibold text-white">Dismiss</Text>
          </TouchableOpacity>
        </View>
      )}

      <ScrollView
        className="flex-1"
        contentContainerStyle={{ flexGrow: 1, paddingTop: insets.top + 24, paddingBottom: 120, paddingHorizontal: 24 }}
        showsVerticalScrollIndicator={false}
      >
        <View className="flex-1 justify-between gap-8">
          <View className="gap-5">
            <Text className="text-5xl font-bold leading-[56px] tracking-tight text-white">
              TrackSense
            </Text>

            <Text className="text-xl leading-8 text-slate-300">
              Tap anywhere on this page to start the robot.
            </Text>

            <Text className="text-lg leading-7 text-slate-400">
              Voice control is always listening. Say TrackGo to start or TrackStop to stop.
            </Text>
          </View>

          <View className="gap-4">
            <View className="rounded-[36px] border border-white/10 bg-slate-900 px-7 py-8 shadow-lg">
              <Text className="text-center text-3xl font-bold leading-10 text-white">Tap Anywhere to Start</Text>
              <Text className="mt-4 text-center text-lg leading-8 text-slate-300">
                Say TrackGo to start or TrackStop to stop.
              </Text>
            </View>
          </View>
        </View>
      </ScrollView>

      <VoiceListeningOverlay
        isListening={isSpeechDetected}
        idleText="Voice control is active. Say TrackGo to start or TrackStop to stop."
        listeningText="Speech detected. Listening for TrackGo or TrackStop."
      />
    </Pressable>
  );
}