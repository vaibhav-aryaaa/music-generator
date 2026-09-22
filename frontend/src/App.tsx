import React, { useState, useEffect, useRef } from "react";
import { 
  Play, 
  Pause, 
  Download, 
  Sparkles, 
  Volume2, 
  VolumeX, 
  RotateCcw, 
  Music
} from "lucide-react";

declare global {
  namespace JSX {
    interface IntrinsicElements {
      'midi-player': React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement> & {
        ref?: React.RefObject<any>;
        src?: string | null;
        "sound-font"?: string;
        style?: React.CSSProperties;
      }, HTMLElement>;
    }
  }
}

interface Note {
  pitch: number;
  time: number;
  duration: number;
  velocity: number;
}

interface StyleCategory {
  id: string;
  name: string;
  meter: string;
  icon: string;
}

const STYLES: StyleCategory[] = [
  { id: "jigs", name: "Jig", meter: "6/8", icon: "🌿" },
  { id: "reels", name: "Reel", meter: "4/4", icon: "🔥" },
  { id: "waltzes", name: "Waltz", meter: "3/4", icon: "🌙" },
  { id: "hornpipes", name: "Hornpipe", meter: "Swung", icon: "🌾" }
];

export default function App() {
  const [mood, setMood] = useState<string>("jigs");
  const [length, setLength] = useState<number>(300);
  const [temp, setTemp] = useState<number>(1.0);
  const [topP, setTopP] = useState<number>(0.9);
  
  const [generating, setGenerating] = useState<boolean>(false);
  const [midiUrl, setMidiUrl] = useState<string | null>(null);
  const [notes, setNotes] = useState<Note[]>([]);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [totalDuration, setTotalDuration] = useState<number>(0);
  const [muted, setMuted] = useState<boolean>(false);

  const playerRef = useRef<any>(null);
  const animationRef = useRef<number | null>(null);
  const visualizerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    return () => {
      if (midiUrl) URL.revokeObjectURL(midiUrl);
    };
  }, [midiUrl]);

  useEffect(() => {
    const playerEl = playerRef.current;
    if (!playerEl) return;

    const handleStart = () => {
      setIsPlaying(true);
      startLoop();
    };

    const handleStop = () => {
      setIsPlaying(false);
      stopLoop();
    };

    playerEl.addEventListener("start", handleStart);
    playerEl.addEventListener("stop", handleStop);
    playerEl.addEventListener("pause", handleStop);

    return () => {
      playerEl.removeEventListener("start", handleStart);
      playerEl.removeEventListener("stop", handleStop);
      playerEl.removeEventListener("pause", handleStop);
      stopLoop();
    };
  }, [midiUrl]);

  const startLoop = () => {
    const update = () => {
      const playerEl = playerRef.current;
      if (playerEl && playerEl.player) {
        const time = playerEl.player.currentTime || 0;
        const dur = playerEl.player.duration || 0;
        setCurrentTime(time);
        if (dur) setTotalDuration(dur);

        // Auto-scroll timeline
        if (visualizerRef.current && dur > 0) {
          const container = visualizerRef.current;
          const svgW = Math.max(600, dur * 60);
          const playheadX = (time / dur) * svgW;
          container.scrollLeft = playheadX - container.clientWidth * 0.3;
        }
      }
      animationRef.current = requestAnimationFrame(update);
    };
    animationRef.current = requestAnimationFrame(update);
  };

  const stopLoop = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  };

  const handleGenerate = async () => {
    setGenerating(true);
    setMidiUrl(null);
    setNotes([]);
    setCurrentTime(0);

    try {
      const res = await fetch("http://127.0.0.1:8000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mood, length, temp, top_p: topP })
      });

      if (!res.ok) throw new Error("Generation failed");

      const data = await res.json();
      const byteChars = atob(data.midi_base64);
      const byteNumbers = new Array(byteChars.length);
      for (let i = 0; i < byteChars.length; i++) {
        byteNumbers[i] = byteChars.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: "audio/midi" });
      const url = URL.createObjectURL(blob);

      setMidiUrl(url);
      setNotes(data.notes || []);

      if (data.notes && data.notes.length > 0) {
        const last = data.notes.reduce((max: Note, n: Note) => 
          (n.time + n.duration > max.time + max.duration ? n : max), data.notes[0]);
        setTotalDuration(last.time + last.duration);
      }
    } catch (err) {
      console.error(err);
      alert("Could not connect to backend server. Make sure it is running on http://127.0.0.1:8000.");
    } finally {
      setGenerating(false);
    }
  };

  const handlePlayPause = () => {
    const playerEl = playerRef.current;
    if (!playerEl) return;
    if (isPlaying) {
      playerEl.pause();
    } else {
      playerEl.start();
    }
  };

  const handleSeek = (newTime: number) => {
    const playerEl = playerRef.current;
    if (!playerEl) return;
    playerEl.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const toggleMute = () => {
    const tone = (window as any).Tone;
    if (tone) {
      tone.Destination.mute = !muted;
    }
    setMuted(!muted);
  };

  const handleDownload = () => {
    if (!midiUrl) return;
    const a = document.createElement("a");
    a.href = midiUrl;
    a.download = `aether_${mood}.mid`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const formatTime = (seconds: number) => {
    if (isNaN(seconds) || seconds <= 0) return "0:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs < 10 ? "0" : ""}${secs}`;
  };

  // Piano roll bounds
  const pitches = notes.length ? notes.map(n => n.pitch) : [60];
  const minPitch = Math.min(...pitches) - 2;
  const maxPitch = Math.max(...pitches) + 2;
  const maxTime = Math.max(...(notes.map(n => n.time + n.duration)), totalDuration, 1);
  const svgWidth = Math.max(600, maxTime * 60);

  const getPitchY = (pitch: number) => {
    return 160 - ((pitch - minPitch) / (maxPitch - minPitch || 1)) * 140;
  };

  return (
    <div className="app" data-theme={mood}>
      {/* Header */}
      <header className="header">
        <div className="logo">
          <Music size={22} className="logo-icon" />
          <h1>Aether</h1>
        </div>
        <p className="subtitle">AI Music Generator</p>
      </header>

      {/* Style Selector */}
      <div className="styles-bar">
        {STYLES.map((s) => (
          <button
            key={s.id}
            className={`style-pill ${mood === s.id ? "active" : ""}`}
            onClick={() => setMood(s.id)}
          >
            <span className="pill-icon">{s.icon}</span>
            <span className="pill-name">{s.name}</span>
            <span className="pill-meter">{s.meter}</span>
          </button>
        ))}
      </div>

      {/* Main Dashboard */}
      <div className="dashboard">
        {/* Controls Card */}
        <div className="card controls-card">
          <h2>Parameters</h2>
          
          <div className="control-item">
            <div className="control-label">
              <span>Length</span>
              <span className="control-val">{length} notes</span>
            </div>
            <input
              type="range"
              min="100"
              max="600"
              step="50"
              value={length}
              onChange={(e) => setLength(parseInt(e.target.value))}
            />
          </div>

          <div className="control-item">
            <div className="control-label">
              <span>Creativity</span>
              <span className="control-val">{temp.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="0.4"
              max="1.5"
              step="0.1"
              value={temp}
              onChange={(e) => setTemp(parseFloat(e.target.value))}
            />
          </div>

          <div className="control-item">
            <div className="control-label">
              <span>Coherence</span>
              <span className="control-val">{topP.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="1.0"
              step="0.05"
              value={topP}
              onChange={(e) => setTopP(parseFloat(e.target.value))}
            />
          </div>

          <button
            className="generate-btn"
            onClick={handleGenerate}
            disabled={generating}
          >
            {generating ? (
              <>
                <div className="spinner"></div>
                Composing...
              </>
            ) : (
              <>
                <Sparkles size={18} />
                Generate Music
              </>
            )}
          </button>
        </div>

        {/* Player & Visualizer Card */}
        <div className="card player-card">
          <div className="player-header">
            <h2>Player</h2>
            {midiUrl && (
              <button className="download-btn" onClick={handleDownload} title="Download MIDI">
                <Download size={15} />
                Download
              </button>
            )}
          </div>

          {midiUrl ? (
            <div className="player-body">
              {/* Visualizer */}
              <div className="visualizer-wrapper" ref={visualizerRef}>
                <svg width={svgWidth} height="180" className="visualizer-svg">
                  {/* Grid Lines */}
                  {[0, 1, 2, 3, 4].map(i => (
                    <line
                      key={`grid-${i}`}
                      x1="0"
                      y1={20 + i * 35}
                      x2={svgWidth}
                      y2={20 + i * 35}
                      className="grid-line"
                    />
                  ))}

                  {/* Notes */}
                  {notes.map((n, i) => {
                    const x = n.time * 60;
                    const w = Math.max(n.duration * 60, 4);
                    const y = getPitchY(n.pitch);
                    const isActive = currentTime >= n.time && currentTime <= n.time + n.duration;
                    return (
                      <rect
                        key={`n-${i}`}
                        x={x}
                        y={y - 4}
                        width={w}
                        height={8}
                        rx={3}
                        className={`note-block ${isActive ? "active" : ""}`}
                      />
                    );
                  })}

                  {/* Playhead */}
                  <line
                    x1={currentTime * 60}
                    y1="0"
                    x2={currentTime * 60}
                    y2="180"
                    className="playhead"
                  />
                </svg>
              </div>

              {/* Playback Controls */}
              <div className="playback-bar">
                <button
                  className={`play-btn ${isPlaying ? "playing" : ""}`}
                  onClick={handlePlayPause}
                >
                  {isPlaying ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" style={{ marginLeft: "2px" }} />}
                </button>

                <span className="time-display">{formatTime(currentTime)}</span>

                <input
                  type="range"
                  min="0"
                  max={totalDuration || 1}
                  step="0.05"
                  value={currentTime}
                  onChange={(e) => handleSeek(parseFloat(e.target.value))}
                  className="seek-bar"
                />

                <span className="time-display">{formatTime(totalDuration)}</span>

                <button
                  className={`mute-btn ${muted ? "muted" : ""}`}
                  onClick={toggleMute}
                  title={muted ? "Unmute" : "Mute"}
                >
                  {muted ? <VolumeX size={17} /> : <Volume2 size={17} />}
                </button>
              </div>

              <midi-player
                ref={playerRef}
                src={midiUrl}
                sound-font="https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus"
                style={{ display: "none" }}
              ></midi-player>
            </div>
          ) : (
            <div className="empty-state">
              <Music size={36} className="empty-icon" />
              <p>Ready to compose. Choose a style and click Generate.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
