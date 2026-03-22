import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ExpoSpeechRecognitionModule,
  useSpeechRecognitionEvent,
} from 'expo-speech-recognition';
import { useTTS } from './useTTS';

const COMMAND_COOLDOWN_MS = 5000;
const TRANSCRIPT_DEDUPE_MS = 1200;
const RESTART_DELAY_MS = 900;
const SPEECH_ACTIVE_RETRY_MS = 900;

export function useVoiceCommand({
  enabled = true,
  alwaysListening = false,
  announceOnStart = true,
  onStartCommand,
  onStopCommand,
} = {}) {
  const { speak, speakWithInterrupt, isSpeaking } = useTTS();
  const [isListening, setIsListening] = useState(false);
  const [isSpeechDetected, setIsSpeechDetected] = useState(false);
  const isListeningRef = useRef(false);
  const pendingStartRef = useRef(false);
  const restartTimeoutRef = useRef(null);
  const lastTranscriptRef = useRef({ text: '', timestamp: 0 });
  const lastCommandAtRef = useRef(0);

  useEffect(() => {
    isListeningRef.current = isListening;
  }, [isListening]);

  const scheduleRestart = useCallback((delay = RESTART_DELAY_MS) => {
    if (restartTimeoutRef.current) {
      clearTimeout(restartTimeoutRef.current);
    }

    restartTimeoutRef.current = setTimeout(async () => {
      if (!alwaysListening || !enabled) {
        return;
      }

      const speaking = await isSpeaking();
      if (speaking) {
        scheduleRestart(SPEECH_ACTIVE_RETRY_MS);
        return;
      }

      startListening(false);
    }, delay);
  }, [alwaysListening, enabled, isSpeaking]);

  const handleTranscript = useCallback(
    (transcript) => {
      if (!enabled || !transcript) {
        return;
      }

      const normalized = transcript.toLowerCase().replace(/[^a-z\s]/g, ' ').replace(/\s+/g, ' ').trim();
      if (!normalized) {
        return;
      }

      const now = Date.now();
      if (
        lastTranscriptRef.current.text === normalized &&
        now - lastTranscriptRef.current.timestamp < TRANSCRIPT_DEDUPE_MS
      ) {
        return;
      }

      lastTranscriptRef.current = { text: normalized, timestamp: now };

      if (now - lastCommandAtRef.current < COMMAND_COOLDOWN_MS) {
        return;
      }

      const hasTrackGo   = normalized.includes('trackgo')   || normalized.includes('track go');
      const hasTrackStop  = normalized.includes('trackstop')  || normalized.includes('track stop');

      if (hasTrackGo) {
        lastCommandAtRef.current = now;
        speakWithInterrupt('Track go command received');
        onStartCommand?.();
        return;
      }

      if (hasTrackStop) {
        lastCommandAtRef.current = now;
        speakWithInterrupt('Track stop command received');
        onStopCommand?.();
        return;
      }
    },
    [enabled, onStartCommand, onStopCommand, speakWithInterrupt]
  );

  useSpeechRecognitionEvent('start', () => {
    pendingStartRef.current = false;
    setIsListening(true);
    setIsSpeechDetected(false);
  });

  useSpeechRecognitionEvent('speechstart', () => {
    setIsSpeechDetected(true);
  });

  useSpeechRecognitionEvent('speechend', () => {
    setIsSpeechDetected(false);
  });

  useSpeechRecognitionEvent('result', (event) => {
    const transcript = (event.results || [])
      .map((result) => result?.transcript || '')
      .join(' ')
      .trim();

    handleTranscript(transcript);
  });

  useSpeechRecognitionEvent('nomatch', () => {
    // Suppressed to prevent spam
  });

  useSpeechRecognitionEvent('error', (event) => {
    pendingStartRef.current = false;
    setIsListening(false);
    setIsSpeechDetected(false);

    if (event?.error === 'aborted') {
      return;
    }

    if (event?.error === 'no-speech') {
      return;
    }

    if (event?.error === 'not-allowed' || event?.error === 'service-not-allowed') {
      speakWithInterrupt('Voice commands are unavailable. Check microphone and speech permissions.');
      return;
    }
  });

  useSpeechRecognitionEvent('end', () => {
    pendingStartRef.current = false;
    setIsListening(false);
    setIsSpeechDetected(false);

    if (alwaysListening && enabled) {
      scheduleRestart(RESTART_DELAY_MS);
    }
  });

  const startListening = useCallback(async (withPrompt = announceOnStart) => {
    if (!enabled || isListeningRef.current || pendingStartRef.current) {
      return;
    }

    try {
      pendingStartRef.current = true;

      if (restartTimeoutRef.current) {
        clearTimeout(restartTimeoutRef.current);
        restartTimeoutRef.current = null;
      }

      if (alwaysListening) {
        const speaking = await isSpeaking();
        if (speaking) {
          pendingStartRef.current = false;
          scheduleRestart(SPEECH_ACTIVE_RETRY_MS);
          return;
        }
      }

      if (!ExpoSpeechRecognitionModule.isRecognitionAvailable()) {
        pendingStartRef.current = false;
        speakWithInterrupt('Speech recognition is not available on this device.');
        return;
      }

      const permission = await ExpoSpeechRecognitionModule.requestPermissionsAsync();
      if (!permission.granted) {
        pendingStartRef.current = false;
        speakWithInterrupt('Microphone permission is required for voice commands.');
        return;
      }

      // Prompts are now handled directly by the homepage/live screens

      ExpoSpeechRecognitionModule.start({
        lang: 'en-US',
        interimResults: true,
        maxAlternatives: 1,
        continuous: alwaysListening,
        requiresOnDeviceRecognition: false,
        contextualStrings: ['TrackGo', 'TrackStop', 'TrackSense'],
        iosTaskHint: 'confirmation',
        iosVoiceProcessingEnabled: true,
        androidIntentOptions: {
          EXTRA_LANGUAGE_MODEL: 'web_search',
        },
      });
    } catch (error) {
      pendingStartRef.current = false;
      setIsListening(false);
      setIsSpeechDetected(false);
      speakWithInterrupt('Unable to start voice recognition.');
      console.error('Speech recognition start failed', error);
    }
  }, [alwaysListening, announceOnStart, enabled, isSpeaking, scheduleRestart, speak, speakWithInterrupt]);

  const stopListening = useCallback(() => {
    if (alwaysListening || !isListening) {
      return;
    }

    try {
      ExpoSpeechRecognitionModule.stop();
    } catch (error) {
      console.error('Speech recognition stop failed', error);
    }
  }, [alwaysListening, isListening]);

  useEffect(() => {
    if (!alwaysListening || !enabled) {
      return undefined;
    }

    startListening(false);

    return () => {
      if (restartTimeoutRef.current) {
        clearTimeout(restartTimeoutRef.current);
        restartTimeoutRef.current = null;
      }
    };
  }, [alwaysListening, enabled, startListening]);

  useEffect(() => {
    return () => {
      if (restartTimeoutRef.current) {
        clearTimeout(restartTimeoutRef.current);
        restartTimeoutRef.current = null;
      }

      try {
        ExpoSpeechRecognitionModule.abort();
      } catch (_) {
        // ignore cleanup abort failures
      }
    };
  }, []);

  return {
    isListening,
    isSpeechDetected,
    startListening,
    stopListening,
  };
}
