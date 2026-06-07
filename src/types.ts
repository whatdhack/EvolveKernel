export interface Problem {
  id: string;
  name: string;
  slug: string;
  description: string;
  gpuDevice: string;
  metric: string;
  seedCode: string;
  inputsDescription: string;
  optimalLeaderboardGflops: number;
  seedGflops: number;
  leaderboard: LeaderboardEntry[];
}

export interface LeaderboardEntry {
  rank: number;
  username: string;
  gflops: number;
  runtimeMs: number;
  date: string;
  isCurrentUser?: boolean;
}

export interface MutationStrategy {
  id: string;
  name: string;
  description: string;
  severity: 'low' | 'medium' | 'high';
}

export interface Candidate {
  id: string;
  generation: number;
  notes: string;
  code: string;
  gflops: number;
  runtimeMs: number;
  compiled: boolean;
  errors?: string;
  mutationType: string;
  isOptimal?: boolean;
}

export interface EvolutionConfig {
  populationSize: number;
  mutationRate: number;
  selectedModel: string;
  strategies: string[];
}

export interface EvolutionStep {
  generation: number;
  candidates: Candidate[];
  reasoning: string;
}

export interface SimulationResult {
  code: string;
  compiled: boolean;
  errors?: string;
  runtimeMs: number;
  gflops: number;
  metrics: {
    coalescence: number; // 0 to 100
    loadStoreEfficiency: number; // 0 to 100
    sharedMemoryOccupancy: number; // 0 to 100
    loopUnrolling: boolean;
    vectorizationFactor: number;
  };
  logOutput: string;
}
