import React, { useState, useEffect, useRef } from "react";
import { 
  Cpu, 
  GitBranch, 
  Github, 
  Play, 
  Check, 
  AlertCircle, 
  RefreshCw, 
  Sliders, 
  Copy, 
  ExternalLink, 
  TrendingUp, 
  Sparkles, 
  Code, 
  Award, 
  Terminal, 
  Settings, 
  Layers, 
  Activity,
  Heart,
  User,
  CheckCircle2,
  Trash2
} from "lucide-react";
import { Problem, Candidate, SimulationResult, EvolutionConfig } from "./types";

export default function App() {
  // Global States
  const [problems, setProblems] = useState<Problem[]>([]);
  const [selectedProblem, setSelectedProblem] = useState<Problem | null>(null);
  const [code, setCode] = useState<string>("");
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null);
  
  // Evolutionary Settings & States
  const [model, setModel] = useState<string>("gemini-3.5-flash");
  const [populationSize, setPopulationSize] = useState<number>(4);
  const [targetGenerations, setTargetGenerations] = useState<number>(3); // Max generations to successive evolve
  const [isEvolving, setIsEvolving] = useState<boolean>(false);
  const [evolutionStep, setEvolutionStep] = useState<number>(0);
  const [evolvedCandidates, setEvolvedCandidates] = useState<Candidate[]>([]);
  const [bestEvolvedGflops, setBestEvolvedGflops] = useState<number>(0);
  const [evolutionReasoning, setEvolutionReasoning] = useState<string>("");
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([
    "coalescing", "vectorization", "unrolling", "block_tuning", "stages_tuning", "warps_tuning", "eviction_policy_tuning", "multiple_of_tuning", "tf32_tuning"
  ]);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  // Submitter Name State
  const [username, setUsername] = useState<string>("Local Optimizer");
  const [githubUser, setGithubUser] = useState<string | null>(null);
  const [githubAvatar, setGithubAvatar] = useState<string | null>(null);

  // Listen for success message from popup (after callback completes)
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      const origin = event.origin;
      if (!origin.endsWith(".run.app") && !origin.includes("localhost")) {
        return;
      }
      if (event.data?.type === "OAUTH_AUTH_SUCCESS") {
        const usernameFromAuth = event.data.username || "devstar2081";
        const avatarUrl = event.data.avatar_url || `https://github.com/${usernameFromAuth}.png`;
        setUsername(usernameFromAuth);
        setGithubUser(usernameFromAuth);
        setGithubAvatar(avatarUrl);
        showToast(`Connected successfully as ${usernameFromAuth}!`, "success");
      }
    };
    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, []);

  const handleConnectGithub = async () => {
    try {
      showToast("Opening GitHub Authorization Screen...", "info");
      const response = await fetch("/api/auth/url");
      if (!response.ok) {
        throw new Error("Failed to fetch auth url");
      }
      const { url } = await response.json();
      
      const authWindow = window.open(
        url,
        "oauth_popup",
        "width=580,height=680"
      );

      if (!authWindow) {
        showToast("Please allow popups to authorize with GitHub.", "info");
      }
    } catch (err) {
      console.error(err);
      showToast("Failed to launch GitHub authentication portal.", "info");
    }
  };

  const handleDisconnectGithub = () => {
    setGithubUser(null);
    setGithubAvatar(null);
    setUsername("Local Optimizer");
    showToast("GitHub account disconnected from Tensara profile integrations.", "info");
  };

  // UI Notification
  const [notification, setNotification] = useState<{message: string, type: 'success' | 'info'} | null>(null);

  // Original Seed Backup
  const [seedRestoreCode, setSeedRestoreCode] = useState<string>("");

  // Refs
  const editorRef = useRef<HTMLTextAreaElement>(null);

  // Fetch initial problems data
  useEffect(() => {
    fetch("/api/problems")
      .then((res) => res.json())
      .then((data) => {
        setProblems(data);
        if (data.length > 0) {
          handleSelectProblem(data[0]);
        }
      })
      .catch((err) => console.error("Failed to load problems:", err));
  }, []);

  const handleSelectProblem = (problem: Problem) => {
    setSelectedProblem(problem);
    setCode(problem.seedCode);
    setSeedRestoreCode(problem.seedCode);
    setSimulationResult(null);
    setEvolvedCandidates([]);
    setBestEvolvedGflops(0);
    setEvolutionStep(0);
    setEvolutionReasoning("");
  };

  const showToast = (message: string, type: 'success' | 'info' = 'success') => {
    setNotification({ message, type });
    setTimeout(() => {
      setNotification(null);
    }, 3000);
  };

  // Run CUDA Sandbox Simulation
  const handleSimulate = async (customCode?: string) => {
    if (!selectedProblem) return;
    setIsSimulating(true);
    try {
      const activeCode = customCode !== undefined ? customCode : code;
      const response = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code: activeCode,
          problemId: selectedProblem.id
        })
      });
      const data = await response.json();
      setSimulationResult(data);
      if (customCode === undefined) {
        // If simulated main editor, update current code speed metric toast
        if (data.compiled) {
          showToast(`Successfully evaluated kernel speed: ${data.gflops} GFLOPS!`, 'success');
        } else {
          showToast(`Compilation warning detected. Review terminal.`, 'info');
        }
      }
    } catch (err) {
      console.error(err);
      showToast("Simulation API Call failed.", "info");
    } finally {
      setIsSimulating(false);
    }
  };

  // Submit directly to Tensara Board and update in-memory leaderboard
  const handleSubmitToTensara = async () => {
    if (!selectedProblem) return;
    setIsSubmitting(true);
    try {
      const response = await fetch("/api/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          problemId: selectedProblem.id,
          code: code,
          username: username || "Local Optimizer"
        })
      });
      const data = await response.json();
      if (!response.ok) {
        showToast(data.error || "Submission failed", "info");
      } else {
        showToast(data.message, "success");
        // Re-fetch problems to get the newly updated leaderboard state
        const res = await fetch("/api/problems");
        const probList = await res.json();
        setProblems(probList);
        const updatedProb = probList.find((p: any) => p.id === selectedProblem.id);
        if (updatedProb) {
          setSelectedProblem(updatedProb);
        }
      }
    } catch (err) {
      console.error(err);
      showToast("Submission system error.", "info");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Launch Server-Side evolutionary logic with multiple successive generations support!
  const handleEvolve = async (gensCount: number = 1) => {
    if (!selectedProblem) return;
    setIsEvolving(true);
    setEvolvedCandidates([]);
    setEvolutionReasoning("");
    setEvolutionStep(0);
    
    try {
      let activeCode = code;
      let currentStepCandidates: Candidate[] = [];
      let finalReasoning = "";

      for (let step = 1; step <= gensCount; step++) {
        setEvolutionStep(step);
        showToast(`Evolving Generation ${step} of ${gensCount}...`, "info");
        
        const response = await fetch("/api/evolve", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            problemId: selectedProblem.id,
            model,
            currentCode: activeCode,
            config: {
              populationSize,
              strategies: selectedStrategies
            }
          })
        });
        
        if (!response.ok) {
          throw new Error(`Evolution server rejected generation count step ${step}`);
        }
        
        const data = await response.json();
        currentStepCandidates = data.candidates || [];
        finalReasoning = data.reasoning || "";
        
        // Take the absolute fastest evolved candidate of this step to seed the next generation
        if (currentStepCandidates.length > 0) {
          currentStepCandidates.sort((a, b) => b.gflops - a.gflops);
          activeCode = currentStepCandidates[0].code;
        }

        // Slight aesthetic delay between successive genetic generations
        if (step < gensCount) {
          await new Promise(resolve => setTimeout(resolve, 800));
        }
      }

      setEvolvedCandidates(currentStepCandidates);
      setEvolutionReasoning(finalReasoning);
      
      const best = currentStepCandidates.length > 0 ? Math.max(...currentStepCandidates.map((c: any) => c.gflops)) : 0;
      setBestEvolvedGflops(best);
      showToast(`Finished evolving ${gensCount} successive generations successfully!`, 'success');
    } catch (err) {
      console.error(err);
      showToast("Evolutionary generation sequence failed.", "info");
    } finally {
      setIsEvolving(false);
    }
  };

  const handleCopyCode = (targetCode: string) => {
    navigator.clipboard.writeText(targetCode);
    showToast("Triton kernel copied to clipboard!", 'success');
  };

  const handleApplyCandidateCode = (cand: Candidate) => {
    setCode(cand.code);
    showToast(`Loaded ${cand.mutationType} candidate to default workspace!`, 'success');
  };

  // Combine customized leaderboard data to incorporate active user
  const getLeaderboardEntries = () => {
    if (!selectedProblem) return [];
    let baseList = [...selectedProblem.leaderboard];
    
    // If the user has a sandbox result, let's inject their current editor speed
    if (simulationResult && simulationResult.compiled) {
      const userGflops = simulationResult.gflops;
      const existsIndex = baseList.findIndex(e => e.isCurrentUser);
      
      const entry = {
        rank: 99, // default placeholder
        username: username ? `${username} (You)` : "Local Optimizer (You)",
        gflops: userGflops,
        runtimeMs: simulationResult.runtimeMs,
        date: "Just now",
        isCurrentUser: true
      };

      if (existsIndex > -1) {
        baseList[existsIndex] = entry;
      } else {
        baseList.push(entry);
      }
    }

    // Sort descending by performance
    baseList.sort((a, b) => b.gflops - a.gflops);
    return baseList.map((entry, idx) => ({ ...entry, rank: idx + 1 }));
  };

  const toggleStrategy = (strategy: string) => {
    if (selectedStrategies.includes(strategy)) {
      setSelectedStrategies(selectedStrategies.filter(s => s !== strategy));
    } else {
      setSelectedStrategies([...selectedStrategies, strategy]);
    }
  };

  const handleRestoreDefault = () => {
    setCode(seedRestoreCode);
    showToast("Workspace template reset to seed reference.", "info");
  };

  return (
    <div className="min-h-screen bg-bg-custom font-sans text-ink">
      
      {/* Toast Notification */}
      {notification && (
        <div className="fixed top-5 right-5 z-50 animate-fade-in flex items-center space-x-3 bg-white border border-emerald-500/25 px-4 py-3 rounded-xl shadow-lg shadow-slate-200/50">
          <div className="h-2 w-2 rounded-full bg-accent animate-pulse" />
          <span className="text-sm font-medium text-slate-800">{notification.message}</span>
        </div>
      )}



      {/* Main Structural Navbar */}
      <header className="border-b border-slate-200 bg-surface/90 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-primary text-white font-extrabold w-8 h-8 rounded-lg flex items-center justify-center text-lg shadow-sm">
              T
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-sans font-semibold text-base text-ink tracking-tight">Tensara Evolve Agent</span>
                <span className="bg-blue-50 text-primary text-[10px] font-semibold py-0.5 px-2 rounded-full border border-blue-200 font-mono">
                  ALPHA-AGENT
                </span>
              </div>
              <p className="text-xs text-secondary">Evolutionary program synthesis & performance sandbox for Tensara GPU Benchmarks</p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            {/* System Status Badges */}
            <div className="hidden md:flex items-center space-x-3">
              <div className="flex items-center space-x-1.5 bg-slate-50 border border-slate-200 px-2.5 py-1 rounded-lg">
                <Activity className="h-3.5 w-3.5 text-rose-500" />
                <span className="text-xs text-secondary font-mono">GPU: <strong className="text-slate-700 font-medium">Tesla T4</strong></span>
              </div>
              <div className="flex items-center space-x-1.5 bg-slate-50 border border-slate-200 px-2.5 py-1 rounded-lg">
                <Sparkles className="h-3.5 w-3.5 text-primary" />
                <span className="text-xs text-secondary font-mono">Synthesis: <strong className="text-slate-700 font-medium">Gemini-3.5</strong></span>
              </div>
            </div>

            {/* Custom Submitter Handle & GitHub Auth */}
            <div className="flex items-center space-x-2">
              {githubUser ? (
                <div className="flex items-center space-x-2 bg-emerald-50 border border-emerald-200 pl-2.5 pr-2 py-1.5 rounded-xl">
                  {githubAvatar ? (
                    <img 
                      src={githubAvatar} 
                      alt={githubUser} 
                      className="h-5 w-5 rounded-full border border-emerald-300"
                      referrerPolicy="no-referrer"
                    />
                  ) : (
                    <Github className="h-4 w-4 text-emerald-600" />
                  )}
                  <span className="text-xs font-mono font-semibold text-emerald-700">{githubUser}</span>
                  <button 
                    onClick={handleDisconnectGithub}
                    className="text-emerald-400 hover:text-red-500 p-0.5 rounded-lg transition-colors cursor-pointer"
                    title="Disconnect GitHub account"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              ) : (
                <button 
                  onClick={handleConnectGithub}
                  className="flex items-center space-x-1.5 bg-[#24292f] hover:bg-[#1f2327] text-white rounded-xl px-3 py-1.5 text-xs font-semibold transition-all shadow-xs cursor-pointer"
                >
                  <Github className="h-4 w-4 text-white" />
                  <span>Sign in with GitHub</span>
                </button>
              )}

              <div className="flex items-center space-x-2 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-xl">
                <User className="h-3.5 w-3.5 text-slate-500" />
                <span className="text-xs text-slate-500 font-medium whitespace-nowrap">Solver Handle:</span>
                <input 
                  type="text" 
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Local Optimizer"
                  className="bg-transparent text-xs font-mono font-bold text-primary focus:outline-none focus:ring-0 w-28 placeholder-slate-400 border-none p-0"
                />
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Layout */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* PROBLEM PICKER & WORKSPACE CONFIGURATION (LEFT - 7 COLUMNS) */}
          <div className="lg:col-span-7 flex flex-col space-y-6">
            
            {/* PROBLEM SELECTOR HEADER */}
            <div className="bg-surface border border-slate-200 rounded-xl p-5 shadow-xs">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <Award className="h-4 w-4 text-primary" />
                  <span className="text-xs font-bold uppercase tracking-wider text-secondary">Select Tensara Problem</span>
                </div>
                {selectedProblem && (
                  <span className="bg-blue-50 text-primary text-xs font-semibold py-1 px-3 rounded-lg border border-blue-200 font-mono">
                    Target: {selectedProblem.optimalLeaderboardGflops} {selectedProblem.id === "matmul" ? "TFLOPS" : "GFLOPS"}
                  </span>
                )}
              </div>

              {/* Selector Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {problems.map((pb) => {
                  const isSelected = selectedProblem?.id === pb.id;
                  return (
                    <button
                      key={pb.id}
                      onClick={() => handleSelectProblem(pb)}
                      className={`text-left p-3 rounded-xl transition-all border cursor-pointer ${
                        isSelected 
                          ? "bg-blue-50/50 border-primary shadow-xs" 
                          : "bg-slate-50/30 border-slate-200 hover:border-slate-300 hover:bg-slate-50"
                      }`}
                    >
                      <h4 className={`text-xs font-bold truncate ${isSelected ? "text-primary" : "text-ink"}`}>
                        {pb.name}
                      </h4>
                      <p className="text-[10px] text-secondary mt-1 uppercase tracking-tight truncate font-medium">
                        {pb.id} • {pb.gpuDevice.replace("NVIDIA ", "")}
                      </p>
                    </button>
                  );
                })}
              </div>

              {selectedProblem && (
                <div className="mt-4 pt-4 border-t border-slate-200 leading-relaxed text-xs text-slate-600">
                  <p className="text-ink font-semibold mb-1">Challenge Summary:</p>
                  <p className="text-slate-500 leading-relaxed">{selectedProblem.description}</p>
                  <p className="text-slate-500 mt-2 font-mono text-[10px]">
                    <strong className="text-primary">Inputs:</strong> {selectedProblem.inputsDescription}
                  </p>
                </div>
              )}
            </div>

            {/* TRITON WORKSPACE & EDITOR */}
            <div className="bg-surface border border-slate-200 rounded-xl flex flex-col overflow-hidden shadow-xs">
              <div className="bg-slate-50/80 border-b border-slate-200 px-4 py-3 flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Code className="h-4 w-4 text-primary" />
                  <span className="text-xs font-bold uppercase tracking-wider text-secondary">Triton Kernel Workspace</span>
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={handleRestoreDefault}
                    className="flex items-center space-x-1.5 text-[10px] text-slate-600 hover:text-ink bg-white hover:bg-slate-50 px-2.5 py-1 rounded-lg border border-slate-200 transition-all font-mono shadow-xs cursor-pointer"
                  >
                    <RefreshCw className="h-3 w-3" />
                    <span>Reset Seed</span>
                  </button>
                  <button
                    onClick={() => handleCopyCode(code)}
                    className="flex items-center space-x-1.5 text-[10px] text-slate-600 hover:text-ink bg-white hover:bg-slate-50 px-2.5 py-1 rounded-lg border border-slate-200 transition-all font-mono shadow-xs cursor-pointer"
                  >
                    <Copy className="h-3 w-3" />
                    <span>Copy</span>
                  </button>
                </div>
              </div>

              {/* Editor Workspace with simulated lines */}
              <div className="flex bg-slate-50/70 font-mono text-xs text-slate-800 min-h-[350px] relative border-b border-slate-200/50">
                {/* Simulated Line Numbers */}
                <div className="bg-slate-100/90 text-slate-400 py-4 px-3 text-right select-none border-r border-slate-200/70 flex flex-col align-middle text-[11px] font-mono w-10">
                  {Array.from({ length: Math.max(15, code.split("\n").length) }, (_, i) => (
                    <div key={i} className="leading-5 h-5">{i + 1}</div>
                  ))}
                </div>
                {/* Editor Textarea */}
                <textarea
                  ref={editorRef}
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  className="flex-1 bg-transparent py-4 px-3 font-mono leading-5 text-slate-800 min-h-[350px] focus:outline-none resize-y"
                  placeholder="# Write your Triton kernel here..."
                  spellCheck="false"
                />
              </div>

              {/* Footer Execution Tools */}
              <div className="bg-slate-50/60 border-t border-slate-200 px-4 py-3.5 flex items-center justify-between">
                <p className="text-[10px] text-secondary font-mono">
                  Target: NVIDIA GPU (Ampere sm_75 / T4)
                </p>
                <button
                  onClick={() => handleSimulate()}
                  disabled={isSimulating}
                  className="flex items-center space-x-1.5 bg-primary hover:bg-blue-700 active:bg-blue-800 disabled:bg-slate-300 text-white font-semibold text-xs py-1.5 px-4 rounded-lg transition-all shadow-xs cursor-pointer"
                >
                  {isSimulating ? (
                    <>
                      <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                      <span>Sandbox Compiling...</span>
                    </>
                  ) : (
                    <>
                      <Play className="h-3.5 w-3.5 fill-current" />
                      <span>Simulate Kernel (T4 Sandbox)</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* WORKSPACE COMPILATION SYSTEM TERMINAL OUTPUT */}
            <div className="bg-slate-900 border border-slate-950 rounded-xl overflow-hidden shadow-inner font-mono text-xs">
              <div className="bg-slate-950 px-4 py-2 flex items-center justify-between border-b border-slate-900">
                <div className="flex items-center space-x-2">
                  <Terminal className="h-4 w-4 text-accent" />
                  <span className="text-[10px] font-medium uppercase tracking-wider text-slate-400 font-mono">T4 CUDA SANDBOX TERMINAL LOG</span>
                </div>
                {simulationResult && (
                  <span className={`text-[10px] font-bold font-mono px-2 py-0.5 rounded-lg border ${
                    simulationResult.compiled 
                      ? "bg-accent/10 text-accent border-accent/20" 
                      : "bg-rose-500/10 text-rose-400 border-rose-500/20"
                  }`}>
                    {simulationResult.compiled ? "SUCCESS" : "ERROR"}
                  </span>
                )}
              </div>
              
              <div className="p-4 font-mono text-[11px] leading-relaxed text-slate-300 max-h-[180px] overflow-y-auto">
                {simulationResult ? (
                  <pre className="whitespace-pre-wrap">{simulationResult.logOutput}</pre>
                ) : (
                  <div className="text-slate-500 italic py-4 text-center">
                    No active sandbox execution recorded. Click "Simulate Kernel (T4 Sandbox)" to run static compliance validation and compiler performance estimates.
                  </div>
                )}
              </div>
            </div>

            {/* PERFORMANCE ANALYSIS PANEL */}
            {simulationResult && simulationResult.compiled && (
              <div className="bg-surface border border-slate-200 rounded-xl p-5 animate-fade-in block shadow-xs">
                <h4 className="text-xs font-bold uppercase tracking-wider text-secondary mb-4 flex items-center space-x-2">
                  <Activity className="h-4 w-4 text-primary" />
                  <span>Interactive Pipeline Metrics Estimator</span>
                </h4>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Coalescing Gauge */}
                  <div className="bg-slate-50 rounded-xl p-3 border border-slate-200">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-[10px] text-slate-500 font-semibold uppercase tracking-tight">Memory Coalescing</span>
                      <span className="text-xs font-mono font-bold text-ink">{simulationResult.metrics.coalescence}%</span>
                    </div>
                    <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                      <div 
                        className="bg-primary h-full rounded-full transition-all" 
                        style={{ width: `${simulationResult.metrics.coalescence}%` }}
                      />
                    </div>
                  </div>

                  {/* Load/Store efficiency */}
                  <div className="bg-slate-50 rounded-xl p-3 border border-slate-200">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-[10px] text-slate-500 font-semibold uppercase tracking-tight">L/S Block Efficiency</span>
                      <span className="text-xs font-mono font-bold text-ink">{simulationResult.metrics.loadStoreEfficiency}%</span>
                    </div>
                    <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                      <div 
                        className="bg-purple-600 h-full rounded-full transition-all" 
                        style={{ width: `${simulationResult.metrics.loadStoreEfficiency}%` }}
                      />
                    </div>
                  </div>

                  {/* Shared occupancy */}
                  <div className="bg-slate-50 rounded-xl p-3 border border-slate-200">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-[10px] text-slate-500 font-semibold uppercase tracking-tight">SM Occupancy (Est.)</span>
                      <span className="text-xs font-mono font-bold text-ink">{simulationResult.metrics.sharedMemoryOccupancy}%</span>
                    </div>
                    <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                      <div 
                        className="bg-indigo-600 h-full rounded-full transition-all" 
                        style={{ width: `${simulationResult.metrics.sharedMemoryOccupancy}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-4 pt-3 border-t border-slate-100 text-[10px] text-slate-500 font-mono">
                  <div>Vectorization: <span className="text-primary font-bold">{simulationResult.metrics.vectorizationFactor}x float aligned</span></div>
                  <div>Loop Unrolling: <span className="text-primary font-bold">{simulationResult.metrics.loopUnrolling ? "ACTIVE" : "NONE"}</span></div>
                  <div>Mean Latency: <span className="text-primary font-bold">{simulationResult.runtimeMs} ms</span></div>
                  <div>Hardware Bounds: <span className="text-rose-600 font-medium font-bold">Compute Bound</span></div>
                </div>
              </div>
            )}

          </div>

          {/* EVOLUTIONARY PROGRAM SYNTHESIS ENGINE (RIGHT - 5 COLUMNS) */}
          <div className="lg:col-span-5 flex flex-col space-y-6">
            
            {/* AGENT CONTROLS & MODEL TUNER */}
            <div className="bg-surface border border-slate-200 rounded-xl p-5 shadow-xs">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <Sliders className="h-4 w-4 text-primary" />
                  <span className="text-xs font-bold uppercase tracking-wider text-secondary">Evolution Engine Settings</span>
                </div>
              </div>

              <div className="space-y-4">
                {/* Models selector */}
                <div>
                  <label className="block text-[11px] uppercase text-slate-500 tracking-wider font-semibold mb-2">
                    Synthesis Model
                  </label>
                  <select 
                    value={model} 
                    onChange={(e) => setModel(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-xs text-ink focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary/40 cursor-pointer text-slate-800"
                  >
                    <option value="gemini-3.5-flash">Gemini 3.5 Flash (Super Fast mutation)</option>
                    <option value="gemini-3.1-pro-preview">Gemini 3.1 Pro (Complex math synthesis)*</option>
                  </select>
                </div>

                {/* Population scale count */}
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <label className="text-[11px] uppercase text-slate-500 tracking-wider font-semibold">
                      Gen Population Pool size
                    </label>
                    <span className="text-xs font-mono font-bold text-primary">{populationSize} candidates / gen</span>
                  </div>
                  <input 
                    type="range" 
                    min={2} 
                    max={6} 
                    value={populationSize} 
                    onChange={(e) => setPopulationSize(parseInt(e.target.value))}
                    className="w-full accent-primary bg-slate-200 rounded-lg appearance-none h-1.5 focus:outline-none cursor-pointer"
                  />
                  <div className="flex justify-between text-[9px] text-slate-400 font-mono mt-1">
                    <span>2 (Fastest)</span>
                    <span>4 (Ideal)</span>
                    <span>6 (Diversified)</span>
                  </div>
                </div>

                {/* Generational Depth Tuning Slider */}
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <label className="text-[11px] uppercase text-indigo-500 tracking-wider font-semibold">
                      Generational Depth Tuning
                    </label>
                    <span className="text-xs font-mono font-bold text-indigo-600">{targetGenerations} consecutive gen{targetGenerations > 1 ? "s" : ""}</span>
                  </div>
                  <input 
                    type="range" 
                    min={1} 
                    max={5} 
                    value={targetGenerations} 
                    onChange={(e) => setTargetGenerations(parseInt(e.target.value))}
                    className="w-full accent-indigo-600 bg-slate-200 rounded-lg appearance-none h-1.5 focus:outline-none cursor-pointer"
                  />
                  <div className="flex justify-between text-[9px] text-slate-400 font-mono mt-1">
                    <span>1 (Single)</span>
                    <span>3 (Moderate)</span>
                    <span>5 (Maximum consecutive recursive synthesis)</span>
                  </div>
                </div>

                {/* Strategy Checkboxes */}
                <div>
                  <label className="block text-[11px] uppercase text-slate-500 tracking-wider font-semibold mb-2">
                    Applied Mutational Focus & Triton Tuning Flags
                  </label>
                  <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
                    <label className="flex items-center space-x-2 bg-slate-50/50 p-2 rounded-lg border border-slate-200 hover:border-slate-300 hover:bg-slate-50 cursor-pointer text-slate-700">
                      <input 
                        type="checkbox" 
                        checked={selectedStrategies.includes("coalescing")} 
                        onChange={() => toggleStrategy("coalescing")}
                        className="accent-primary h-3.5 w-3.5 cursor-pointer animate-none"
                      />
                      <span>Coalesce Ops</span>
                    </label>
                    <label className="flex items-center space-x-2 bg-slate-50/50 p-2 rounded-lg border border-slate-200 hover:border-slate-300 hover:bg-slate-50 cursor-pointer text-slate-700">
                      <input 
                        type="checkbox" 
                        checked={selectedStrategies.includes("vectorization")} 
                        onChange={() => toggleStrategy("vectorization")}
                        className="accent-primary h-3.5 w-3.5 cursor-pointer"
                      />
                      <span>Vector Loading</span>
                    </label>
                    <label className="flex items-center space-x-2 bg-slate-50/50 p-2 rounded-lg border border-slate-200 hover:border-slate-300 hover:bg-slate-50 cursor-pointer text-slate-700">
                      <input 
                        type="checkbox" 
                        checked={selectedStrategies.includes("unrolling")} 
                        onChange={() => toggleStrategy("unrolling")}
                        className="accent-primary h-3.5 w-3.5 cursor-pointer"
                      />
                      <span>Loop Unroll</span>
                    </label>
                    <label className="flex items-center space-x-2 bg-slate-50/50 p-2 rounded-lg border border-slate-200 hover:border-slate-300 hover:bg-slate-50 cursor-pointer text-slate-700">
                      <input 
                        type="checkbox" 
                        checked={selectedStrategies.includes("block_tuning")} 
                        onChange={() => toggleStrategy("block_tuning")}
                        className="accent-primary h-3.5 w-3.5 cursor-pointer"
                      />
                      <span>Block Constexpr</span>
                    </label>
                    <label className="flex items-center space-x-2 bg-slate-50/50 p-2 rounded-lg border border-slate-200 hover:border-slate-300 hover:bg-slate-50 cursor-pointer text-slate-705">
                      <input 
                        type="checkbox" 
                        checked={selectedStrategies.includes("stages_tuning")} 
                        onChange={() => toggleStrategy("stages_tuning")}
                        className="accent-indigo-600 h-3.5 w-3.5 cursor-pointer"
                      />
                      <span>Pipelining (num_stages)</span>
                    </label>
                    <label className="flex items-center space-x-2 bg-slate-50/50 p-2 rounded-lg border border-slate-200 hover:border-slate-300 hover:bg-slate-50 cursor-pointer text-slate-705">
                      <input 
                        type="checkbox" 
                        checked={selectedStrategies.includes("warps_tuning")} 
                        onChange={() => toggleStrategy("warps_tuning")}
                        className="accent-indigo-600 h-3.5 w-3.5 cursor-pointer"
                      />
                      <span>Warps Count (num_warps)</span>
                    </label>
                    <label className="flex items-center space-x-2 bg-slate-50/50 p-2 rounded-lg border border-slate-200 hover:border-slate-300 hover:bg-slate-50 cursor-pointer text-slate-705">
                      <input 
                        type="checkbox" 
                        checked={selectedStrategies.includes("eviction_policy_tuning")} 
                        onChange={() => toggleStrategy("eviction_policy_tuning")}
                        className="accent-indigo-600 h-3.5 w-3.5 cursor-pointer"
                      />
                      <span>Eviction (evict_last)</span>
                    </label>
                    <label className="flex items-center space-x-2 bg-slate-50/50 p-2 rounded-lg border border-slate-200 hover:border-slate-300 hover:bg-slate-50 cursor-pointer text-slate-705">
                      <input 
                        type="checkbox" 
                        checked={selectedStrategies.includes("multiple_of_tuning")} 
                        onChange={() => toggleStrategy("multiple_of_tuning")}
                        className="accent-indigo-600 h-3.5 w-3.5 cursor-pointer"
                      />
                      <span>Alignment (multiple_of)</span>
                    </label>
                    <label className="flex items-center space-x-2 bg-slate-50/50 p-2 rounded-lg border border-slate-200 hover:border-slate-300 hover:bg-slate-50 cursor-pointer text-slate-705 col-span-2">
                      <input 
                        type="checkbox" 
                        checked={selectedStrategies.includes("tf32_tuning")} 
                        onChange={() => toggleStrategy("tf32_tuning")}
                        className="accent-indigo-600 h-3.5 w-3.5 cursor-pointer"
                      />
                      <span>Tensor Core Float Precision (allow_tf32)</span>
                    </label>
                  </div>
                </div>

                {/* Main Launch Action */}
                <button
                  onClick={() => handleEvolve(targetGenerations)}
                  disabled={isEvolving || !selectedProblem}
                  className="w-full flex items-center justify-center space-x-2 bg-primary hover:bg-blue-700 disabled:bg-slate-300 text-white font-semibold text-sm py-2.5 px-4 rounded-lg transition-all shadow-xs cursor-pointer"
                >
                  {isEvolving ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin text-white" />
                      <span>Evolving generation sequence ({evolutionStep > 0 ? `${evolutionStep}/${targetGenerations}` : "1..."})...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4 text-white fill-current animate-pulse" />
                      <span>Synthesize & Evolve {targetGenerations} Gen{targetGenerations > 1 ? "s" : ""}</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* GENERATED LINEAGE & CANDIDATES POOL */}
            {evolutionStep > 0 && (
              <div className="bg-surface border border-slate-200 rounded-xl p-5 shadow-xs space-y-4 animate-fade-in block">
                <div className="flex justify-between items-center pb-2 border-b border-slate-100">
                  <div className="flex items-center space-x-2">
                    <GitBranch className="h-4 w-4 text-purple-600" />
                    <span className="text-xs font-bold uppercase tracking-wider text-secondary">Generation Lineage ({evolutionStep})</span>
                  </div>
                  {bestEvolvedGflops > 0 && (
                    <span className="bg-purple-50 text-purple-600 border border-purple-200 font-mono text-[10px] font-bold px-2 py-0.5 rounded-lg">
                      Best Evolved Max: {bestEvolvedGflops} {selectedProblem?.id === "matmul" ? "TFLOPS" : "GFLOPS"}
                    </span>
                  )}
                </div>

                {/* Synthesis reasoning note */}
                <p className="text-[11px] text-slate-600 italic bg-slate-50/80 p-3 rounded-xl border border-slate-200/50 font-sans leading-relaxed">
                  "{evolutionReasoning}"
                </p>

                {/* Candidates List */}
                <div className="space-y-3">
                  {evolvedCandidates.map((cand, idx) => {
                    const isOptimal = cand.gflops >= (selectedProblem?.optimalLeaderboardGflops || 99999);
                    return (
                      <div 
                        key={cand.id} 
                        className={`bg-slate-50 border ${
                          isOptimal 
                          ? "border-emerald-500 bg-emerald-50/30 text-slate-800" 
                          : "border-slate-200/80"
                        } rounded-xl p-3 flex flex-col justify-between`}
                      >
                        <div className="flex justify-between items-start mb-2">
                          <div>
                            <span className="bg-white border border-slate-200 px-2 py-0.5 rounded-md font-mono text-[9px] uppercase font-bold text-slate-600">
                              {cand.mutationType}
                            </span>
                            <span className="ml-2 font-mono text-[10px] text-slate-400">Candidate #{idx + 1}</span>
                          </div>
                          <span className={`font-mono text-xs font-bold ${isOptimal ? "text-accent" : "text-primary"}`}>
                            {cand.gflops} {selectedProblem?.id === "matmul" ? "TFLOPS" : "GFLOPS"}
                          </span>
                        </div>

                        <p className="text-[10px] text-slate-500 leading-normal mb-3">
                          {cand.notes}
                        </p>

                        <div className="flex space-x-2">
                          <button
                            onClick={() => handleApplyCandidateCode(cand)}
                            className="flex-1 bg-white hover:bg-slate-100 border border-slate-200 text-[10px] font-mono text-slate-700 py-1.5 rounded-lg transition-colors cursor-pointer shadow-xs"
                          >
                            Apply to Editor
                          </button>
                          <button
                            onClick={() => handleSimulate(cand.code)}
                            className="bg-blue-50/50 hover:bg-blue-100/50 text-[10px] border border-blue-200 text-primary font-mono py-1.5 px-3 rounded-lg transition-colors cursor-pointer"
                          >
                            Sim Validate
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* LEADERBOARD VIEW AND SUBMISSION WIDGET */}
            <div className="bg-surface border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
              <div className="flex justify-between items-center pb-2 border-b border-slate-100">
                <div className="flex items-center space-x-2">
                  <Award className="h-4 w-4 text-amber-500" />
                  <span className="text-xs font-bold uppercase tracking-wider text-secondary">Tensara.org Leaderboard</span>
                </div>
                {selectedProblem && (
                  <a 
                    href={`https://tensara.org/problems/${selectedProblem.slug}`}
                    target="_blank" 
                    rel="noreferrer"
                    className="flex items-center space-x-1 font-mono text-[10px] text-primary hover:text-blue-700 transition-colors"
                  >
                    <span>View Official</span>
                    <ExternalLink className="h-3 w-3" />
                  </a>
                )}
              </div>

              {/* Leaderboard Table */}
              <div className="space-y-1.5">
                {getLeaderboardEntries().map((entry) => {
                  const isUser = entry.isCurrentUser;
                  const isTopSeat = entry.rank === 1;

                  return (
                    <div 
                      key={`${entry.username}_${entry.rank}`}
                      className={`flex items-center justify-between p-2 rounded-xl transition-all border ${
                        isUser 
                          ? "bg-blue-50/50 border-l-4 border-l-primary border-slate-200 shadow-sm" 
                          : "bg-slate-50/30 hover:bg-slate-50/80 border-transparent"
                      }`}
                    >
                      <div className="flex items-center space-x-2.5 truncate">
                        <span className={`font-mono text-xs font-bold w-4 text-center ${
                          isTopSeat ? "text-amber-500 text-sm" : isUser ? "text-primary" : "text-slate-400"
                        }`}>
                          {entry.rank}
                        </span>
                        <div className="truncate">
                          <h5 className={`text-xs ${isUser ? "text-primary font-semibold" : isTopSeat ? "text-ink font-semibold" : "text-slate-700"} truncate`}>
                            {entry.username}
                          </h5>
                          <span className="text-[9px] text-slate-400 font-mono">
                            {entry.date} • {entry.runtimeMs} ms
                          </span>
                        </div>
                      </div>

                      <div className="text-right">
                        <span className={`font-mono text-xs font-bold ${
                          isUser ? "text-primary" : isTopSeat ? "text-amber-500" : "text-slate-800"
                        }`}>
                          {entry.gflops}
                        </span>
                        <p className="text-[9px] text-slate-400 uppercase tracking-tight font-medium">Gflops</p>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Live Direct Submission Module */}
              <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl flex flex-col space-y-3 mt-4">
                <div className="flex items-start space-x-2.5">
                  <CheckCircle2 className="h-4.5 w-4.5 text-blue-600 shrink-0 mt-0.5" />
                  <div className="text-xs text-slate-650">
                    <p className="text-ink font-semibold">Tensara Leaderboard Submission</p>
                    <p className="text-slate-500 mt-1 leading-normal">
                      Submit and benchmark your code through our compiler engine. Claim your seat rank on the verified simulation leaderboard.
                    </p>
                  </div>
                </div>

                <div className="flex items-center justify-between text-[11px] px-2.5 py-1.5 rounded-lg bg-white border border-slate-200">
                  <span className="text-slate-500">Submitter Identity:</span>
                  {githubUser ? (
                    <div className="flex items-center space-x-1 font-mono font-bold text-emerald-700">
                      <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                      <span>@{githubUser} (GitHub Verified)</span>
                    </div>
                  ) : (
                    <span className="font-mono font-bold text-amber-700">@{username} (Unverified Sandbox)</span>
                  )}
                </div>

                <div className="flex flex-col space-y-2 pt-1">
                  {/* Button for simulation submissions */}
                  <button
                    onClick={handleSubmitToTensara}
                    disabled={isSubmitting}
                    className="w-full flex items-center justify-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white text-xs py-2 rounded-lg transition-all font-semibold shadow-xs cursor-pointer"
                  >
                    {isSubmitting ? (
                      <>
                        <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                        <span>Uploading Kernel to Tensara...</span>
                      </>
                    ) : (
                      <>
                        <Award className="h-3.5 w-3.5 text-white/90" />
                        <span>Submit to Tensara Board</span>
                      </>
                    )}
                  </button>

                  <div className="border-t border-slate-200/60 my-1"></div>

                  <p className="text-[10px] text-slate-500 text-center italic">
                    To place on the official global live <strong>Tensara.org</strong> board:
                  </p>

                  <div className="flex space-x-2">
                    <button
                      onClick={() => handleCopyCode(code)}
                      className="flex-1 flex items-center justify-center space-x-1.5 bg-emerald-50 hover:bg-emerald-100 text-emerald-700 text-xs py-2 rounded-lg border border-emerald-200 transition-all font-mono font-semibold cursor-pointer"
                    >
                      <Copy className="h-3.5 w-3.5" />
                      <span>1-Click Copy Code</span>
                    </button>
                    {selectedProblem && (
                      <a
                        href={`https://tensara.org/problems/${selectedProblem.slug}`}
                        target="_blank"
                        rel="noreferrer"
                        className="flex-1 flex items-center justify-center space-x-1 bg-white hover:bg-slate-50 text-slate-650 text-xs px-3 py-1.5 rounded-lg border border-slate-200 transition-all font-mono shadow-3xs"
                      >
                        <span>Paste on Tensara</span>
                        <ExternalLink className="h-3 w-3 hover:scale-105" />
                      </a>
                    )}
                  </div>
                </div>
              </div>
            </div>

          </div>

        </div>
      </main>

      {/* Humble simple Footer */}
      <footer className="border-t border-slate-200 bg-white mt-12 py-8 text-secondary">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col tracking-tight md:flex-row items-center justify-between text-xs">
          <p>© 2026 TritonEvolve. Built dynamically for high-performance compiler evaluation.</p>
          <div className="flex space-x-4 mt-4 md:mt-0 font-mono text-slate-400">
            <span>Server Active: Port 3000</span>
            <span>GPU Sandbox: Tesla T4</span>
          </div>
        </div>
      </footer>

    </div>
  );
}
