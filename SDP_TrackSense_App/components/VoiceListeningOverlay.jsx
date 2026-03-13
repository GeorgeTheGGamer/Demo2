import { useEffect, useRef } from 'react';
import { Animated, Easing, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

export function VoiceListeningOverlay({ isListening, idleText, listeningText }) {
  const overlayOpacity = useRef(new Animated.Value(0)).current;
  const micTranslateY = useRef(new Animated.Value(48)).current;
  const micOpacity = useRef(new Animated.Value(0)).current;
  const micScale = useRef(new Animated.Value(0.92)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(overlayOpacity, {
        toValue: isListening ? 1 : 0,
        duration: 260,
        easing: Easing.out(Easing.ease),
        useNativeDriver: true,
      }),
      Animated.timing(micTranslateY, {
        toValue: isListening ? 0 : 48,
        duration: 260,
        easing: Easing.out(Easing.ease),
        useNativeDriver: true,
      }),
      Animated.timing(micOpacity, {
        toValue: isListening ? 1 : 0,
        duration: 220,
        easing: Easing.out(Easing.ease),
        useNativeDriver: true,
      }),
      Animated.timing(micScale, {
        toValue: isListening ? 1 : 0.92,
        duration: 240,
        easing: Easing.out(Easing.ease),
        useNativeDriver: true,
      }),
    ]).start();
  }, [isListening, micOpacity, micScale, micTranslateY, overlayOpacity]);

  return (
    <>
      <Animated.View
        pointerEvents="none"
        style={{ opacity: overlayOpacity }}
        className="absolute inset-0 bg-sky-500/25"
      />

      <View pointerEvents="none" className="absolute bottom-8 left-0 right-0 items-center px-6">
        <Animated.View
          style={{
            opacity: micOpacity,
            transform: [{ translateY: micTranslateY }, { scale: micScale }],
          }}
          className="mb-4 items-center rounded-[28px] border border-sky-300/40 bg-sky-500/92 px-7 py-6"
        >
          <Ionicons name="mic" size={36} color="white" />
          <Text className="mt-3 text-center text-lg font-semibold text-white">{listeningText}</Text>
        </Animated.View>

        {!isListening && <Text className="text-center text-base text-slate-300">{idleText}</Text>}
      </View>
    </>
  );
}
