import { useCallback, useRef } from 'react';
import * as Speech from 'expo-speech';

/**
 * useTTS — Text-to-Speech hook for the TrackSense accessibility layer.
 *
 * speak(text)              Queue speech. Will not cut off what is already playing.
 * speakWithInterrupt(text) Stop current speech immediately, then speak. Use for
 *                          dangers and auto-stop events.
 */
export function useTTS() {
  const urgentLockUntilRef = useRef(0);

  const speak = useCallback((text) => {
    if (!text) return;

    if (Date.now() < urgentLockUntilRef.current) {
      return;
    }

    Speech.speak(text, {
      language: 'en-US',
      pitch: 1.0,
      rate: 0.9,
    });
  }, []);

  const speakWithInterrupt = useCallback((text, cooldownMs = 3500) => {
    if (!text) return;

    urgentLockUntilRef.current = Date.now() + cooldownMs;
    Speech.stop();
    Speech.speak(text, {
      language: 'en-US',
      pitch: 1.0,
      rate: 0.9,
    });
  }, []);

  const isSpeaking = useCallback(async () => {
    try {
      return await Speech.isSpeakingAsync();
    } catch (_) {
      return false;
    }
  }, []);

  return { speak, speakWithInterrupt, isSpeaking };
}
