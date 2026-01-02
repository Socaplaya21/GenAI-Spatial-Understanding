
import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage } from '@google/genai';
import { BoundingBox, TranscriptionEntry, ConnectionStatus } from './types';
import { decode, encode, decodeAudioData, createPcmBlob, downsample } from './utils/audio-utils';
import BoundingBoxOverlay from './components/BoundingBoxOverlay';

const MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';
const SYSTEM_INSTRUCTION = `
You are a spatial reasoning expert. 
Your task is to detect objects in the video stream and provide their locations using normalized bounding boxes.
A bounding box is a list of four numbers: [ymin, xmin, ymax, xmax] where each number is between 0 and 1000.
Always mention the labels of the objects you find and follow it with their box coordinates.
Example: "I see a coffee cup at [200, 300, 450, 500] and a laptop at [600, 100, 950, 800]."
Focus on responding naturally to the user while performing this visual detection.
Keep your verbal responses concise and always prioritize accuracy in coordinates.
`;

const App: React.FC = () => {
  // UI State
  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.DISCONNECTED);
  const [history, setHistory] = useState<TranscriptionEntry[]>([]);
  const [detectedBoxes, setDetectedBoxes] = useState<BoundingBox[]>([]);
  const [videoRatio, setVideoRatio] = useState<number>(16/9); 
  
  // Refs
  const sessionRef = useRef<any>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const inputAudioCtxRef = useRef<AudioContext | null>(null);
  const outputAudioCtxRef = useRef<AudioContext | null>(null);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const nextStartTimeRef = useRef<number>(0);
  const frameIntervalRef = useRef<number | null>(null);
  const historyEndRef = useRef<HTMLDivElement>(null);
  const processingBufferRef = useRef<string>('');

  // Audio Processing State
  const currentOutputTranscription = useRef<string>('');
  const currentInputTranscription = useRef<string>('');

  const scrollToBottom = () => {
    historyEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [history]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopSession();
    };
  }, []);

  // Prune old boxes periodically
  useEffect(() => {
    const interval = setInterval(() => {
      setDetectedBoxes(prev => {
        const now = Date.now();
        const active = prev.filter(b => now - b.timestamp < 3000);
        if (active.length !== prev.length) return active;
        return prev;
      });
    }, 500);
    return () => clearInterval(interval);
  }, []);

  const handleVideoLoad = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const video = e.currentTarget;
    if (video.videoWidth && video.videoHeight) {
      setVideoRatio(video.videoWidth / video.videoHeight);
    }
  };

  const calculateDistance = (box1: BoundingBox, box2: Partial<BoundingBox>) => {
    const c1x = (box1.xmin + box1.xmax) / 2;
    const c1y = (box1.ymin + box1.ymax) / 2;
    const c2x = ((box2.xmin || 0) + (box2.xmax || 0)) / 2;
    const c2y = ((box2.ymin || 0) + (box2.ymax || 0)) / 2;
    return Math.sqrt(Math.pow(c1x - c2x, 2) + Math.pow(c1y - c2y, 2));
  };

  const processTextForBoxes = (newText: string) => {
    // Append new text to a small buffer to handle split tokens
    processingBufferRef.current += newText;
    
    // Keep buffer size manageable (last 500 chars should catch any split box def)
    if (processingBufferRef.current.length > 500) {
      processingBufferRef.current = processingBufferRef.current.slice(-500);
    }

    const textToScan = processingBufferRef.current;
    
    // Regex for standard [ymin, xmin, ymax, xmax] format (0-1000)
    const boxRegex = /([a-zA-Z\s]+)?\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]/g;
    const foundBoxes: Partial<BoundingBox>[] = [];
    let match;
    
    // Find all matches in the current buffer
    while ((match = boxRegex.exec(textToScan)) !== null) {
      const rawLabel = match[1]?.trim();
      // Heuristic: If label is empty, look back in text? For now default to Object if missing.
      const label = (rawLabel || 'Object').split(' ').pop() || 'Object'; 
      
      const ymin = parseInt(match[2]);
      const xmin = parseInt(match[3]);
      const ymax = parseInt(match[4]);
      const xmax = parseInt(match[5]);

      if (!isNaN(ymin) && !isNaN(xmin) && !isNaN(ymax) && !isNaN(xmax)) {
        foundBoxes.push({
          label,
          ymin,
          xmin,
          ymax,
          xmax,
          timestamp: Date.now()
        });
      }
    }

    if (foundBoxes.length > 0) {
      setDetectedBoxes(prevBoxes => {
        const nextBoxes = [...prevBoxes];
        
        foundBoxes.forEach(newBox => {
          // Attempt to match with existing box
          let bestMatchIndex = -1;
          let minDistance = 200; // Threshold for tracking (20% of screen)

          nextBoxes.forEach((existingBox, idx) => {
            if (existingBox.label.toLowerCase() === newBox.label?.toLowerCase()) {
              const dist = calculateDistance(existingBox, newBox);
              if (dist < minDistance) {
                minDistance = dist;
                bestMatchIndex = idx;
              }
            }
          });

          if (bestMatchIndex !== -1) {
            // Update existing box
            nextBoxes[bestMatchIndex] = {
              ...nextBoxes[bestMatchIndex],
              ...newBox,
              id: nextBoxes[bestMatchIndex].id, // Preserve ID for CSS transition
              timestamp: Date.now()
            };
          } else {
            // Add new box
            nextBoxes.push({
              id: Math.random().toString(36).substr(2, 9),
              label: newBox.label || 'Object',
              ymin: newBox.ymin!,
              xmin: newBox.xmin!,
              ymax: newBox.ymax!,
              xmax: newBox.xmax!,
              timestamp: Date.now()
            });
          }
        });

        return nextBoxes;
      });
    }
  };

  const startSession = async () => {
    try {
      setStatus(ConnectionStatus.CONNECTING);
      
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 }, 
          height: { ideal: 720 }, 
          frameRate: { ideal: 15 } 
        }, 
        audio: true 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      const inputCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const outputCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      
      inputAudioCtxRef.current = inputCtx;
      outputAudioCtxRef.current = outputCtx;
      
      const outputNode = outputCtx.createGain();
      outputNode.connect(outputCtx.destination);

      const sessionPromise = ai.live.connect({
        model: MODEL_NAME,
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: SYSTEM_INSTRUCTION,
          outputAudioTranscription: {},
          inputAudioTranscription: {},
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } }
          }
        },
        callbacks: {
          onopen: () => {
            console.log('Gemini Live session opened');
            setStatus(ConnectionStatus.CONNECTED);

            if (inputAudioCtxRef.current) {
              const source = inputAudioCtxRef.current.createMediaStreamSource(stream);
              const scriptProcessor = inputAudioCtxRef.current.createScriptProcessor(4096, 1, 1);
              const nativeSampleRate = inputAudioCtxRef.current.sampleRate;

              scriptProcessor.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                const downsampledData = downsample(inputData, nativeSampleRate, 16000);
                const pcmBlob = createPcmBlob(downsampledData);
                
                sessionPromise.then((session) => {
                  session.sendRealtimeInput({ media: pcmBlob });
                });
              };
              source.connect(scriptProcessor);
              scriptProcessor.connect(inputAudioCtxRef.current.destination);
            }

            frameIntervalRef.current = window.setInterval(() => {
              if (canvasRef.current && videoRef.current) {
                const video = videoRef.current;
                const canvas = canvasRef.current;
                const ctx = canvas.getContext('2d');
                
                if (ctx && video.videoWidth > 0) {
                  if (canvas.width !== video.videoWidth) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                  }
                  
                  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                  canvas.toBlob((blob) => {
                    if (blob) {
                      const reader = new FileReader();
                      reader.onloadend = () => {
                        const base64Data = (reader.result as string).split(',')[1];
                        sessionPromise.then((session) => {
                          session.sendRealtimeInput({
                            media: { data: base64Data, mimeType: 'image/jpeg' }
                          });
                        });
                      };
                      reader.readAsDataURL(blob);
                    }
                  }, 'image/jpeg', 0.85);
                }
              }
            }, 500);
          },
          onmessage: async (message: LiveServerMessage) => {
            if (message.serverContent?.outputTranscription) {
              const text = message.serverContent.outputTranscription.text;
              currentOutputTranscription.current += text;
              processTextForBoxes(text);
            } else if (message.serverContent?.inputTranscription) {
              currentInputTranscription.current += message.serverContent.inputTranscription.text;
            }

            if (message.serverContent?.turnComplete) {
              if (currentInputTranscription.current) {
                setHistory(prev => [...prev, { role: 'user', text: currentInputTranscription.current }]);
                currentInputTranscription.current = '';
              }
              if (currentOutputTranscription.current) {
                setHistory(prev => [...prev, { role: 'model', text: currentOutputTranscription.current }]);
                currentOutputTranscription.current = '';
                // Clear buffer on turn complete to reset context? 
                // processingBufferRef.current = ''; 
              }
            }

            const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (base64Audio && outputAudioCtxRef.current) {
              const ctx = outputAudioCtxRef.current;
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);
              const audioBuffer = await decodeAudioData(decode(base64Audio), ctx, 24000, 1);
              const source = ctx.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(outputNode);
              source.addEventListener('ended', () => {
                sourcesRef.current.delete(source);
              });
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += audioBuffer.duration;
              sourcesRef.current.add(source);
            }

            if (message.serverContent?.interrupted) {
              sourcesRef.current.forEach(s => {
                try { s.stop(); } catch(e) {}
              });
              sourcesRef.current.clear();
              nextStartTimeRef.current = 0;
            }
          },
          onerror: (e) => {
            console.error('Gemini Live Error:', e);
            setStatus(ConnectionStatus.ERROR);
          },
          onclose: () => {
            setStatus(ConnectionStatus.DISCONNECTED);
            stopSession();
          }
        }
      });

      sessionRef.current = await sessionPromise;

    } catch (err) {
      console.error('Failed to start session:', err);
      setStatus(ConnectionStatus.ERROR);
    }
  };

  const stopSession = () => {
    if (frameIntervalRef.current) {
      window.clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    if (sessionRef.current) {
      sessionRef.current.close();
      sessionRef.current = null;
    }
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    sourcesRef.current.forEach(s => {
      try { s.stop(); } catch(e) {}
    });
    sourcesRef.current.clear();
    setDetectedBoxes([]);
    setStatus(ConnectionStatus.DISCONNECTED);
  };

  return (
    <div className="flex flex-col lg:flex-row h-screen w-full bg-slate-950 text-slate-100 overflow-hidden font-sans">
      <div className="flex-1 flex flex-col p-6 space-y-4 min-h-0">
        <header className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className={`w-2.5 h-2.5 rounded-full ${status === ConnectionStatus.CONNECTED ? 'bg-green-500 shadow-[0_0_10px_#22c55e]' : status === ConnectionStatus.CONNECTING ? 'bg-yellow-500 animate-pulse' : 'bg-red-500'}`} />
            <h1 className="text-xl font-black tracking-tighter uppercase italic">Gemini <span className="text-cyan-400">Spatial</span></h1>
          </div>
          <div className="text-[10px] text-slate-500 uppercase tracking-[0.3em] font-bold bg-slate-900 px-4 py-1.5 rounded-full border border-slate-800">
            {status}
          </div>
        </header>

        <div className="relative flex-1 flex items-center justify-center min-h-0">
          <div 
            className="relative w-full max-w-5xl bg-black rounded-3xl border border-slate-800 overflow-hidden shadow-2xl flex items-center justify-center transition-all duration-500"
            style={{ aspectRatio: videoRatio }}
          >
            <div className="relative w-full h-full scale-x-[-1]">
              <video 
                ref={videoRef} 
                onLoadedMetadata={handleVideoLoad}
                autoPlay 
                playsInline 
                muted 
                className="w-full h-full object-fill"
              />
              <canvas ref={canvasRef} className="hidden" />
              <BoundingBoxOverlay boxes={detectedBoxes} />
            </div>

            <div className="absolute top-6 right-6 pointer-events-none">
              <div className="bg-black/60 backdrop-blur-xl border border-white/10 px-4 py-2 rounded-2xl flex items-center space-x-3 shadow-2xl">
                <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                <span className="text-[10px] font-mono text-white/70 uppercase tracking-widest">Live HD Feed</span>
              </div>
            </div>

            <div className="absolute bottom-6 left-6 flex flex-wrap gap-2 max-w-[80%] pointer-events-none">
              {detectedBoxes.length > 0 && (
                <div className="bg-cyan-500 text-slate-950 px-4 py-1.5 rounded-full text-[10px] font-black uppercase tracking-[0.2em] shadow-xl animate-bounce">
                  Grounding: {detectedBoxes.length} Targets
                </div>
              )}
            </div>

            {status === ConnectionStatus.DISCONNECTED && (
              <div className="absolute inset-0 bg-slate-950/90 backdrop-blur-md flex flex-col items-center justify-center space-y-8 z-30 scale-x-100">
                <div className="text-center space-y-3 px-8">
                  <h2 className="text-3xl font-black uppercase italic tracking-tight">Vision Interface</h2>
                  <p className="text-slate-400 max-w-xs mx-auto text-sm font-medium leading-relaxed">
                    Connect to activate high-precision spatial awareness and object tracking.
                  </p>
                </div>
                <button 
                  onClick={startSession}
                  className="group relative px-10 py-4 bg-white text-slate-950 font-black uppercase tracking-widest text-xs rounded-full transition-all hover:scale-105 active:scale-95 overflow-hidden"
                >
                  <span className="relative z-10">Initialize Session</span>
                  <div className="absolute inset-0 bg-cyan-400 translate-y-full group-hover:translate-y-0 transition-transform duration-300" />
                </button>
              </div>
            )}

            {status === ConnectionStatus.CONNECTING && (
              <div className="absolute inset-0 bg-slate-950/90 backdrop-blur-md flex flex-col items-center justify-center space-y-6 z-30 scale-x-100">
                <div className="relative">
                  <div className="w-16 h-16 border-2 border-white/10 rounded-full" />
                  <div className="absolute inset-0 w-16 h-16 border-t-2 border-cyan-400 rounded-full animate-spin" />
                </div>
                <p className="text-cyan-400 font-mono text-[10px] tracking-[0.4em] uppercase font-bold">Synchronizing Sensors</p>
              </div>
            )}
          </div>
        </div>

        <div className="h-12 flex items-center justify-center">
          {status === ConnectionStatus.CONNECTED && (
            <button 
              onClick={stopSession}
              className="px-6 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-500 text-[10px] font-black rounded-full transition-all uppercase tracking-[0.2em]"
            >
              Terminate
            </button>
          )}
        </div>
      </div>

      <div className="w-full lg:w-96 border-l border-slate-900 bg-slate-900/30 backdrop-blur-3xl flex flex-col h-1/3 lg:h-full">
        <div className="p-6 border-b border-slate-900 bg-slate-900/50">
          <h2 className="font-black text-slate-400 text-[10px] uppercase tracking-[0.4em]">Neural Output</h2>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth custom-scrollbar">
          {history.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-700 space-y-6 opacity-30">
              <div className="w-px h-12 bg-gradient-to-b from-transparent to-slate-700" />
              <p className="text-[10px] uppercase tracking-[0.3em] font-bold">Awaiting Stream</p>
            </div>
          ) : (
            history.map((item, idx) => (
              <div 
                key={idx} 
                className={`flex flex-col space-y-2 animate-in fade-in slide-in-from-bottom-4 duration-500 ${
                  item.role === 'user' ? 'items-end' : 'items-start'
                }`}
              >
                <div className={`text-[8px] font-black uppercase tracking-widest ${
                  item.role === 'user' ? 'text-slate-500' : 'text-cyan-500'
                }`}>
                  {item.role === 'user' ? 'Operator' : 'AI Core'}
                </div>
                <div className={`max-w-[85%] p-4 rounded-2xl text-xs leading-relaxed font-medium ${
                  item.role === 'user' 
                    ? 'bg-slate-800 text-slate-200 rounded-tr-none' 
                    : 'bg-cyan-500/5 border border-cyan-500/10 text-cyan-50 rounded-tl-none'
                }`}>
                  {item.text}
                </div>
              </div>
            ))
          )}
          <div ref={historyEndRef} />
        </div>

        <div className="p-8 bg-slate-950/50">
          <div className="bg-slate-900/50 p-4 rounded-2xl border border-slate-800/50">
            <p className="text-[10px] text-slate-500 text-center leading-loose font-medium italic">
              "Find my headphones" or "What's in front of me?"
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
